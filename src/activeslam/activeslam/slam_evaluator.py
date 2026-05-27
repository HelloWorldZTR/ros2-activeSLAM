import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import GetEntityState
from nav_msgs.msg import OccupancyGrid, Odometry
try:
    from rclpy.clock import Clock, ClockType
except ImportError:  # pragma: no cover - ROS distro compatibility.
    from rclpy.clock import Clock
    from rclpy.clock_type import ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener

from .slam_evaluator_utils import (
    Bounds,
    Pose2D,
    accumulate_path_length,
    compute_ate,
    compute_coverage,
    compute_map_iou,
    derive_bounds_from_obstacles,
    extract_box_obstacles,
    rasterize_obstacles,
    yaw_from_quaternion,
)

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - depends on runtime display/backend.
    plt = None


class SlamEvaluator(Node):
    def __init__(self):
        super().__init__('slam_evaluator')

        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        self.world_name = self.declare_parameter('world_name', 'slam_rooms').value
        self.log_root = self.declare_parameter('log_root', 'logs').value
        self.sample_interval = self.declare_parameter('sample_interval', 1.0).value
        self.map_topic = self.declare_parameter('map_topic', '/map').value
        self.est_parent_frame = self.declare_parameter('est_parent_frame', 'map').value
        self.est_child_frame = self.declare_parameter(
            'est_child_frame',
            'base_footprint',
        ).value
        self.est_child_frame_candidates = self._parse_csv_parameter(
            self.declare_parameter(
                'est_child_frame_candidates',
                'base_footprint,base_link',
            ).value
        )
        self.gt_topic = self.declare_parameter(
            'gt_topic',
            '/model_states',
        ).value
        self.gt_topic_candidates = self._parse_csv_parameter(
            self.declare_parameter(
                'gt_topic_candidates',
                '/model_states,/gazebo/model_states',
            ).value
        )
        self.gt_service = self.declare_parameter(
            'gt_service',
            '/get_entity_state',
        ).value
        self.gt_odom_topic = self.declare_parameter(
            'gt_odom_topic',
            '/odom',
        ).value
        self.gt_model_name = self.declare_parameter(
            'gt_model_name',
            os.environ.get('TURTLEBOT3_MODEL', 'burger'),
        ).value
        self.eval_margin = self.declare_parameter('eval_margin', 0.5).value
        self.eval_min_x = self.declare_parameter('eval_min_x', float('nan')).value
        self.eval_max_x = self.declare_parameter('eval_max_x', float('nan')).value
        self.eval_min_y = self.declare_parameter('eval_min_y', float('nan')).value
        self.eval_max_y = self.declare_parameter('eval_max_y', float('nan')).value
        self.plot_live = self._as_bool(
            self.declare_parameter('plot_live', True).value
        )
        self.save_plots = self._as_bool(
            self.declare_parameter('save_plots', True).value
        )

        self.world_name = self._normalize_world_name(self.world_name)
        self.world_path = self._resolve_world_path(self.world_name)
        self.obstacles = extract_box_obstacles(str(self.world_path))
        self.eval_bounds = self._resolve_eval_bounds()

        self.run_dir = self._create_run_dir()
        self.csv_files = {}
        self.csv_writers = {}
        self._open_csv_files()
        self._flush_csv_files()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self._map_callback,
            10,
        )
        self.gt_subs = []
        for topic in self._gt_topic_candidates():
            self.gt_subs.append(
                self.create_subscription(
                    ModelStates,
                    topic,
                    lambda msg, topic=topic: self._gt_callback(msg, topic),
                    10,
                )
            )
        self.gt_service_client = self.create_client(
            GetEntityState,
            self.gt_service,
        )
        self.gt_odom_sub = self.create_subscription(
            Odometry,
            self.gt_odom_topic,
            self._gt_odom_callback,
            10,
        )

        self.start_time = self._now_sec()
        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_gt: Optional[Tuple[float, float, float]] = None
        self.latest_gt_timestamp: Optional[float] = None
        self.latest_gt_topic: Optional[str] = None
        self.gt_model_candidates = self._build_gt_model_candidates()
        self.latest_gt_model_names = []
        self.last_gt_names_log_time = 0.0
        self.gt_service_future = None
        self.gt_service_candidate_index = 0
        self.latest_est_available = False
        self.latest_est_frame: Optional[str] = None
        self.est_samples = []
        self.gt_samples = []
        self.coverage_times = []
        self.coverage_values = []
        self.path_lengths = []
        self.ate_times = []
        self.ate_values = []
        self.total_path_length = 0.0
        self.previous_est_xy: Optional[Tuple[float, float]] = None
        self.final_coverage: Optional[float] = None
        self.finalized = False
        self.last_status_log_time = 0.0
        self.ready_logged = False

        self.fig = None
        self.axes = None
        self.lines = {}
        self._init_plotting()

        self.sample_timer = self.create_timer(
            float(self.sample_interval),
            self._sample,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

        self.get_logger().info(
            f'Evaluator logging to {self.run_dir} for world {self.world_name}'
        )
        self.get_logger().info(
            f'Sampling every {self.sample_interval:.2f}s; waiting for map/TF data.'
        )

    def _normalize_world_name(self, world_name: str) -> str:
        name = str(world_name).strip()
        if name.endswith('.world'):
            name = name[:-6]
        if not name.startswith('slam') or '/' in name or '\\' in name:
            raise ValueError('world_name must be a slam*.world basename')
        return name

    def _resolve_world_path(self, world_name: str) -> Path:
        resource_dir = Path(get_package_share_directory('activeslam_resource'))
        world_path = resource_dir / 'maps' / f'{world_name}.world'
        if not world_path.exists():
            raise FileNotFoundError(f'World file not found: {world_path}')
        return world_path

    def _resolve_eval_bounds(self) -> Optional[Bounds]:
        manual_values = (
            self.eval_min_x,
            self.eval_max_x,
            self.eval_min_y,
            self.eval_max_y,
        )
        if all(math.isfinite(float(v)) for v in manual_values):
            min_x, max_x, min_y, max_y = (float(v) for v in manual_values)
            return min_x, max_x, min_y, max_y

        bounds = derive_bounds_from_obstacles(self.obstacles, float(self.eval_margin))
        if bounds is None:
            self.get_logger().warning(
                'No inline box obstacles found; coverage will use full map extent.'
            )
        return bounds

    def _create_run_dir(self) -> Path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = Path(self.log_root).expanduser() / f'run_{timestamp}'
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _open_csv_files(self):
        self._open_csv(
            'trajectory_est',
            ('time_sec', 'x', 'y', 'yaw'),
        )
        self._open_csv(
            'trajectory_gt',
            ('time_sec', 'x', 'y', 'yaw'),
        )
        self._open_csv(
            'coverage_time',
            ('time_sec', 'coverage', 'known_cells', 'total_cells'),
        )
        self._open_csv(
            'coverage_path',
            ('path_length', 'coverage'),
        )

    def _open_csv(self, name: str, header):
        handle = open(self.run_dir / f'{name}.csv', 'w', newline='')
        writer = csv.writer(handle)
        writer.writerow(header)
        self.csv_files[name] = handle
        self.csv_writers[name] = writer

    def _map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg

    def _gt_callback(self, msg: ModelStates, topic: str):
        self.latest_gt_model_names = list(msg.name)
        index = self._find_gt_model_index(msg.name)
        if index is None:
            return

        pose = msg.pose[index]
        yaw = yaw_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        self.latest_gt = (pose.position.x, pose.position.y, yaw)
        self.latest_gt_timestamp = self._now_sec()
        self.latest_gt_topic = topic

    def _gt_odom_callback(self, msg: Odometry):
        if self.latest_gt is not None and self.latest_gt_topic != self.gt_odom_topic:
            return

        pose = msg.pose.pose
        yaw = yaw_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        self.latest_gt = (pose.position.x, pose.position.y, yaw)
        self.latest_gt_timestamp = self._now_sec()
        self.latest_gt_topic = self.gt_odom_topic

    def _build_gt_model_candidates(self):
        configured = str(self.gt_model_name)
        candidates = [configured]

        if configured.startswith('turtlebot3_'):
            candidates.append(configured.removeprefix('turtlebot3_'))
        else:
            candidates.append(f'turtlebot3_{configured}')

        for name in ('burger', 'waffle', 'waffle_pi'):
            candidates.append(name)
            candidates.append(f'turtlebot3_{name}')

        deduped = []
        for name in candidates:
            if name and name not in deduped:
                deduped.append(name)
        return deduped

    def _find_gt_model_index(self, model_names):
        for candidate in self.gt_model_candidates:
            if candidate in model_names:
                if candidate != self.gt_model_name:
                    self.gt_model_name = candidate
                    self.get_logger().info(
                        f'Using Gazebo ground-truth model: {candidate}'
                    )
                return model_names.index(candidate)

        normalized_candidates = {
            candidate.removeprefix('turtlebot3_')
            for candidate in self.gt_model_candidates
        }
        for index, model_name in enumerate(model_names):
            normalized_name = str(model_name).removeprefix('turtlebot3_')
            if normalized_name in normalized_candidates:
                if model_name != self.gt_model_name:
                    self.gt_model_name = model_name
                    self.get_logger().info(
                        f'Using Gazebo ground-truth model: {model_name}'
                    )
                return index
        return None

    def _gt_topic_candidates(self):
        topics = [self.gt_topic] + list(self.gt_topic_candidates)
        deduped = []
        for topic in topics:
            topic = str(topic).strip()
            if topic and topic not in deduped:
                deduped.append(topic)
        return deduped

    def _request_gt_service_pose(self):
        if self.gt_service_future is not None and not self.gt_service_future.done():
            return
        if not self.gt_service_client.service_is_ready():
            self.gt_service_client.wait_for_service(timeout_sec=0.0)
            if not self.gt_service_client.service_is_ready():
                return

        candidates = self.gt_model_candidates
        if not candidates:
            return
        self.gt_service_candidate_index %= len(candidates)
        model_name = candidates[self.gt_service_candidate_index]
        request = GetEntityState.Request()
        request.name = model_name
        request.reference_frame = 'world'
        future = self.gt_service_client.call_async(request)
        self.gt_service_future = future
        future.add_done_callback(
            lambda done, name=model_name: self._gt_service_callback(done, name)
        )

    def _gt_service_callback(self, future, model_name: str):
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().debug(f'Ground-truth service unavailable: {exc}')
            return
        finally:
            self.gt_service_future = None

        if not response.success:
            self.gt_service_candidate_index += 1
            return

        pose = response.state.pose
        yaw = yaw_from_quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        self.latest_gt = (pose.position.x, pose.position.y, yaw)
        self.latest_gt_timestamp = self._now_sec()
        self.latest_gt_topic = f'{self.gt_service}:{model_name}'
        if model_name != self.gt_model_name:
            self.gt_model_name = model_name
            self.get_logger().info(
                f'Using Gazebo ground-truth entity service model: {model_name}'
            )

    def _sample(self):
        now_sec = self._now_sec()
        elapsed = now_sec - self.start_time

        self._request_gt_service_pose()
        est_pose = self._lookup_estimated_pose()
        self.latest_est_available = est_pose is not None
        if est_pose is not None:
            x, y, yaw = est_pose
            self.total_path_length, self.previous_est_xy = accumulate_path_length(
                self.previous_est_xy,
                (x, y),
                self.total_path_length,
            )
            est_sample = Pose2D(elapsed, x, y, yaw)
            self.est_samples.append(est_sample)
            self.csv_writers['trajectory_est'].writerow((elapsed, x, y, yaw))

        if self.latest_gt is not None:
            gt_x, gt_y, gt_yaw = self.latest_gt
            gt_sample = Pose2D(elapsed, gt_x, gt_y, gt_yaw)
            self.gt_samples.append(gt_sample)
            self.csv_writers['trajectory_gt'].writerow(
                (elapsed, gt_x, gt_y, gt_yaw)
            )

        if self.latest_map is not None:
            coverage, known_cells, total_cells = self._coverage_for_latest_map()
            self.final_coverage = coverage
            self.coverage_times.append(elapsed)
            self.coverage_values.append(coverage)
            self.path_lengths.append(self.total_path_length)
            self.csv_writers['coverage_time'].writerow(
                (elapsed, coverage, known_cells, total_cells)
            )
            self.csv_writers['coverage_path'].writerow(
                (self.total_path_length, coverage)
            )

        if self.est_samples and self.gt_samples:
            _, ate_errors = compute_ate(self.est_samples, self.gt_samples)
            if ate_errors:
                self.ate_times = [timestamp for timestamp, _ in ate_errors]
                self.ate_values = [error for _, error in ate_errors]

        self._flush_csv_files()
        self._write_metrics_json(self._base_metrics())
        self._update_plots()
        self._log_status_if_needed()

    def _coverage_for_latest_map(self) -> Tuple[float, int, int]:
        info = self.latest_map.info
        return compute_coverage(
            self.latest_map.data,
            info.width,
            info.height,
            info.resolution,
            info.origin.position.x,
            info.origin.position.y,
            self.eval_bounds,
        )

    def _lookup_estimated_pose(self) -> Optional[Tuple[float, float, float]]:
        child_frames = self._estimated_child_frame_candidates()
        last_error = None
        for child_frame in child_frames:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.est_parent_frame,
                    child_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.2),
                )
            except Exception as exc:
                last_error = exc
                continue

            if child_frame != self.est_child_frame:
                self.est_child_frame = child_frame
                self.get_logger().info(
                    f'Using estimated pose TF: {self.est_parent_frame}->{child_frame}'
                )
            self.latest_est_frame = child_frame
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            yaw = yaw_from_quaternion(rotation.x, rotation.y, rotation.z, rotation.w)
            return translation.x, translation.y, yaw

        if last_error is not None:
            self.get_logger().debug(f'Estimated pose unavailable: {last_error}')
        return None

    def _estimated_child_frame_candidates(self):
        frames = [self.est_child_frame] + list(self.est_child_frame_candidates)
        deduped = []
        for frame in frames:
            frame = str(frame).strip()
            if frame and frame not in deduped:
                deduped.append(frame)
        return deduped

    @staticmethod
    def _parse_csv_parameter(value):
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return [part.strip() for part in str(value).split(',') if part.strip()]

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

    def _init_plotting(self):
        if not self.plot_live:
            return

        if plt is None:
            self.get_logger().warning('matplotlib is unavailable; live plots disabled.')
            return

        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=(8, 9))
        self.lines['coverage_time'], = self.axes[0].plot([], [], 'b-')
        self.lines['coverage_path'], = self.axes[1].plot([], [], 'g-')
        self.lines['ate'], = self.axes[2].plot([], [], 'r-')

        self.axes[0].set_xlabel('Time [s]')
        self.axes[0].set_ylabel('Coverage')
        self.axes[0].set_title('Coverage vs. Time')
        self.axes[1].set_xlabel('Path Length [m]')
        self.axes[1].set_ylabel('Coverage')
        self.axes[1].set_title('Coverage vs. Path Length')
        self.axes[2].set_xlabel('Time [s]')
        self.axes[2].set_ylabel('ATE [m]')
        self.axes[2].set_title('ATE Over Time')
        self.fig.tight_layout()
        self.fig.show()

    def _update_plots(self):
        if plt is None or self.fig is None:
            return

        self.lines['coverage_time'].set_data(
            self.coverage_times,
            self.coverage_values,
        )
        self.lines['coverage_path'].set_data(
            self.path_lengths,
            self.coverage_values,
        )
        self.lines['ate'].set_data(self.ate_times, self.ate_values)

        for axis in self.axes:
            axis.relim()
            axis.autoscale_view()

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def _save_final_outputs(self):
        metrics = self._base_metrics()

        if self.latest_map is not None:
            self._save_occupancy_map(self.latest_map)
            occupied_iou, free_iou = self._compute_final_iou(self.latest_map)
            metrics['occupied_iou'] = occupied_iou
            metrics['free_iou'] = free_iou
        else:
            metrics['occupied_iou'] = None
            metrics['free_iou'] = None

        self._write_metrics_json(metrics)

        if self.save_plots and plt is not None:
            if self.fig is not None:
                self.fig.savefig(self.run_dir / 'metrics_live_plots.png', dpi=150)
            self._save_metric_plot(
                'coverage_time.png',
                self.coverage_times,
                self.coverage_values,
                'Time [s]',
                'Coverage',
                'Coverage vs. Time',
            )
            self._save_metric_plot(
                'coverage_path.png',
                self.path_lengths,
                self.coverage_values,
                'Path Length [m]',
                'Coverage',
                'Coverage vs. Path Length',
            )
            if self.ate_times and self.ate_values:
                self._save_metric_plot(
                    'ate_time.png',
                    self.ate_times,
                    self.ate_values,
                    'Time [s]',
                    'ATE [m]',
                    'ATE Over Time',
                )

    def _base_metrics(self):
        metrics = {
            'final_coverage': self.final_coverage,
            'total_path_length': self.total_path_length,
            'total_time': self._now_sec() - self.start_time,
            'world_name': self.world_name,
            'evaluation_bounds': self.eval_bounds,
            'gt_model_name': self.gt_model_name,
            'gt_topic': self.latest_gt_topic or self.gt_topic,
            'est_parent_frame': self.est_parent_frame,
            'est_child_frame': self.latest_est_frame or self.est_child_frame,
            'estimated_samples': len(self.est_samples),
            'ground_truth_samples': len(self.gt_samples),
        }

        ate_rmse, ate_errors = compute_ate(self.est_samples, self.gt_samples)
        if ate_rmse is not None:
            metrics['ate_rmse'] = ate_rmse
            metrics['ate_samples'] = len(ate_errors)

        return metrics

    def _write_metrics_json(self, metrics):
        tmp_path = self.run_dir / 'metrics.json.tmp'
        with open(tmp_path, 'w') as handle:
            json.dump(metrics, handle, indent=2)
        tmp_path.replace(self.run_dir / 'metrics.json')

    def _compute_final_iou(
        self,
        grid: OccupancyGrid,
    ) -> Tuple[Optional[float], Optional[float]]:
        info = grid.info
        gt_occupied = rasterize_obstacles(
            self.obstacles,
            info.width,
            info.height,
            info.resolution,
            info.origin.position.x,
            info.origin.position.y,
        )
        return compute_map_iou(
            grid.data,
            gt_occupied,
            info.width,
            info.height,
            info.resolution,
            info.origin.position.x,
            info.origin.position.y,
            self.eval_bounds,
        )

    def _save_occupancy_map(self, grid: OccupancyGrid):
        image_path = self.run_dir / 'final_map.pgm'
        yaml_path = self.run_dir / 'final_map.yaml'
        data = np.array(grid.data, dtype=np.int16).reshape(
            grid.info.height,
            grid.info.width,
        )
        image = np.full(data.shape, 205, dtype=np.uint8)
        image[data == 0] = 254
        image[data >= 50] = 0

        # OccupancyGrid row 0 is the map's lower y row; PGM row 0 is top.
        image = np.flipud(image)
        with open(image_path, 'wb') as handle:
            handle.write(f'P5\n{grid.info.width} {grid.info.height}\n255\n'.encode())
            handle.write(image.tobytes())

        origin = grid.info.origin
        yaw = yaw_from_quaternion(
            origin.orientation.x,
            origin.orientation.y,
            origin.orientation.z,
            origin.orientation.w,
        )
        yaml_text = (
            'image: final_map.pgm\n'
            f'resolution: {grid.info.resolution}\n'
            f'origin: [{origin.position.x}, {origin.position.y}, {yaw}]\n'
            'negate: 0\n'
            'occupied_thresh: 0.65\n'
            'free_thresh: 0.196\n'
        )
        yaml_path.write_text(yaml_text)

    def _save_metric_plot(
        self,
        filename: str,
        xs,
        ys,
        xlabel: str,
        ylabel: str,
        title: str,
    ):
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.plot(xs, ys)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        fig.tight_layout()
        fig.savefig(self.run_dir / filename, dpi=150)
        plt.close(fig)

    def _flush_csv_files(self):
        for handle in self.csv_files.values():
            handle.flush()
            os.fsync(handle.fileno())

    def _log_status_if_needed(self):
        now = time.monotonic()
        if now - self.last_status_log_time < 5.0:
            return
        self.last_status_log_time = now

        missing = []
        if self.latest_map is None:
            missing.append(self.map_topic)
        if not self.latest_est_available:
            missing.append(
                f'TF {self.est_parent_frame}->{self.est_child_frame}'
            )
        if self.latest_gt is None:
            missing.append(
                f'ground truth model {self.gt_model_name} on '
                + ', '.join(self._gt_topic_candidates())
                + f', {self.gt_service}, or {self.gt_odom_topic}'
            )

        if missing:
            self.get_logger().info(
                'Waiting for data: ' + ', '.join(missing)
            )
            if self.latest_gt is None and self.latest_gt_model_names:
                names = ', '.join(self.latest_gt_model_names[:12])
                if len(self.latest_gt_model_names) > 12:
                    names += ', ...'
                self.get_logger().info(
                    f'Gazebo models currently published: {names}'
                )
            return

        if not self.ready_logged:
            self.ready_logged = True
            self.get_logger().info('Evaluator is recording samples.')

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def destroy_node(self):
        if not self.finalized:
            self.finalized = True
            self._save_final_outputs()
            for handle in self.csv_files.values():
                handle.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SlamEvaluator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
