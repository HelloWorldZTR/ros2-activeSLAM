# Active SLAM Problem Formulation

本文根据当前代码整理主动 SLAM 的统一符号、通用问题定义，以及各
`slam_mode` 方法在同一套符号下的简要描述。这里的“SLAM 方法”主要指
探索目标选择策略；`slam_toolbox` 负责在线建图，Nav2 负责路径规划、控制、
恢复行为和 `/cmd_vel`，`exploration_coordinator` 负责观测、候选生成、目标选择
和 action 调度。

## 1. 统一符号

### 1.1 时间、状态与地图

令离散决策时刻为

$$
t = 0,1,2,\ldots
$$

机器人在 `map` 坐标系下的位姿为

$$
\mathbf{x}_t = (x_t, y_t, \theta_t) \in SE(2).
$$

`slam_toolbox` 发布的占据栅格记为

$$
\mathcal{M}_t : \Omega_t \rightarrow \{-1,0,\ldots,100\},
$$

其中 $\Omega_t \subset \mathbb{Z}^2$ 是当前 `/map` 数组的栅格集合。
本文约定

$$
\mathcal{U}_t = \{c \in \Omega_t \mid \mathcal{M}_t(c) = -1\},
$$

$$
\mathcal{F}^{\mathrm{free}}_t = \{c \in \Omega_t \mid \mathcal{M}_t(c) = 0\},
$$

$$
\mathcal{O}_t = \{c \in \Omega_t \mid \mathcal{M}_t(c) > 50\}.
$$

世界坐标与栅格坐标之间的映射记为

$$
\pi_t : \mathbb{R}^2 \rightarrow \Omega_t,
\qquad
\pi_t^{-1} : \Omega_t \rightarrow \mathbb{R}^2.
$$

### 1.2 Frontier 与 Safe Goal

frontier cluster 集合记为

$$
\mathcal{C}_t = \mathcal{C}^{\mathrm{unk}}_t \cup
\mathcal{C}^{\mathrm{edge}}_t.
$$

其中 $\mathcal{C}^{\mathrm{unk}}_t$ 是 unknown frontier 聚类，
$\mathcal{C}^{\mathrm{edge}}_t$ 是 compact `/map` 边界上的 open-edge frontier 聚类。
对每个 cluster $C \in \mathcal{C}_t$，代码不会直接导航到 cluster 质心，而是在已知
free 区域内搜索安全落点

$$
\mathbf{g}(C) \in \mathbb{R}^2.
$$

安全落点必须满足已知 free、obstacle clearance、地图边缘约束、Nav2 到达半径约束，
并且从 frontier seed 到落点的短线段不能穿过 occupied cell。若找不到安全落点，
则 $\mathbf{g}(C)=\varnothing$。

frontier 的局部潜在信息增益记为

$$
I_t(C) =
\mathrm{area}\left(
\mathcal{U}_t \cap B(\mathbf{p}_C, r_I)
\right),
$$

其中 $\mathbf{p}_C$ 是 cluster 代表点，$r_I$ 对应
`frontier_information_gain_radius`。对 open-edge frontier，圆形邻域中落在
`/map` 数组外的区域也计为潜在 unknown。

所有可用 frontier 候选组成共享候选池

$$
\mathcal{A}^{F}_t =
\left\{
a_C = (C, \mathbf{g}(C), I_t(C))
\mid C \in \mathcal{C}_t,\ \mathbf{g}(C) \neq \varnothing
\right\}.
$$

基础 frontier utility 为

$$
U_F(a_C \mid \mathbf{x}_t)
=
\frac{I_t(C)}
{\left\|\mathbf{g}(C)-\mathbf{p}(\mathbf{x}_t)\right\|_2 + \epsilon_d},
\qquad
\epsilon_d = 0.1.
$$

其中 $\mathbf{p}(\mathbf{x}_t)=(x_t,y_t)$。

### 1.3 Nav2 可达路径与动作代价

Nav2 对候选目标 $\mathbf{q}$ 的路径预检记为

$$
P_t(\mathbf{q}) =
\mathrm{Nav2Path}(\mathbf{x}_t,\mathbf{q}; \mathcal{M}_t).
$$

若 Nav2 无法给出可执行路径，则

$$
P_t(\mathbf{q}) = \varnothing.
$$

可达候选集合记为

$$
\mathcal{R}_t(\mathcal{A})
=
\{a \in \mathcal{A} \mid P_t(\mathbf{q}(a)) \neq \varnothing\},
$$

其中 $\mathbf{q}(a)$ 是候选 $a$ 对应的 Nav2 goal。路径长度记为

$$
L(P_t) =
\sum_{i=1}^{n}
\left\|
\mathbf{p}_i-\mathbf{p}_{i-1}
\right\|_2.
$$

目标失败或超时后，代码会把目标附近区域加入冷却集合

$$
\mathcal{B}_t \subset \mathbb{R}^2,
$$

后续候选若落入 $\mathcal{B}_t$，会被暂时过滤。

### 1.4 近似 Pose Graph 与 D-opt 指标

探索层维护的近似 pose graph 记为

$$
\mathcal{G}^{P}_t =
\left(\mathcal{V}^{P}_t,\mathcal{E}^{P}_t\right),
$$

其中节点来自 TF 轨迹采样，边包含 odometry-like edge 和几何近似 loop edge。
每条边 $e$ 的信息矩阵为

$$
\boldsymbol{\Omega}_e \in \mathbb{R}^{3 \times 3}.
$$

代码把信息矩阵的正特征值几何均值作为标量边权

$$
w_e =
\exp
\left(
\frac{1}{|\Lambda_e^+|}
\sum_{\lambda \in \Lambda_e^+}
\log \lambda
\right),
$$

并构造加权图 Laplacian

$$
\mathbf{L}_{ij} =
\begin{cases}
\sum_{e \sim i} w_e, & i=j,\\
-w_{ij}, & i \neq j \text{ 且 } (i,j)\in\mathcal{E}^{P}_t,\\
0, & \text{otherwise}.
\end{cases}
$$

锚定第一个节点后得到 reduced Laplacian $\mathbf{L}^{\star}$。近似 D-opt 分数为

$$
D(\mathcal{G}^{P}_t)
=
n^{1/n}
\exp
\left(
\frac{1}{n}
\log \det \mathbf{L}^{\star}
\right),
$$

其中 $n=|\mathcal{V}^{P}_t|$。这是探索层的启发式评分，不是
`slam_toolbox` 后端内部的真实 pose graph。

### 1.5 拓扑图、GVD 图与局部 Region

通用拓扑图记为

$$
\mathcal{G}^{T}_t =
\left(\mathcal{V}^{T}_t,\mathcal{E}^{T}_t\right),
$$

每个顶点 $v$ 有世界坐标 $\mathbf{z}_v \in \mathbb{R}^2$，每条边有长度代价

$$
\ell_e > 0.
$$

GBSAE 使用 per-world 先验图 $\mathcal{G}^{B}$。GVD 方法使用在线构建的
live GVD 图 $\mathcal{G}^{G}_t$。两者都可作为 $\mathcal{G}^{T}_t$ 的实例。

对分层 GVD，宏观顶点 $v$ 可绑定一个局部清扫区域

$$
\mathcal{Q}_t(v) \subset \Omega_t.
$$

局部区域的已观测比例为

$$
\rho_t(v)
=
\frac{
\left|
\{c \in \mathcal{Q}_t(v) \mid \mathcal{M}_t(c) \neq -1\}
\right|
}{
\left|\mathcal{Q}_t(v)\right|
}.
$$

当 $\rho_t(v)$ 超过阈值，或该 Region 内没有 Nav2 可达 frontier 时，该局部清扫完成。

## 2. 通用 Active SLAM Formulation

当前实现可抽象为一个 receding-horizon 的主动观测决策问题。每个决策时刻，
系统根据当前 belief map、机器人位姿、失败冷却集合以及可选的辅助图结构，选择下一个
Nav2 目标或行为：

$$
a_t^\star
=
\arg\max_{a \in \mathcal{R}_t(\mathcal{A}_t)}
J_m(a \mid \mathcal{S}_t),
$$

其中 $m$ 是 `slam_mode`，$\mathcal{A}_t$ 是该模式生成的候选集合，
$\mathcal{S}_t$ 是当前探索状态：

$$
\mathcal{S}_t =
\left(
\mathcal{M}_t,\mathbf{x}_t,\mathcal{B}_t,
\mathcal{G}^{P}_t,\mathcal{G}^{T}_t,\Xi_t
\right).
$$

$\Xi_t$ 表示方法特有的离散状态，例如 GBSAE route index、GVD phase、已探索顶点、
已清扫顶点、当前局部 Region 等。

约束条件由 Nav2 和安全候选生成共同定义：

$$
\mathbf{q}(a_t^\star)
\in
\mathcal{X}^{\mathrm{safe}}_t,
\qquad
P_t(\mathbf{q}(a_t^\star)) \neq \varnothing,
\qquad
\mathbf{q}(a_t^\star) \notin \mathcal{B}_t.
$$

探索的长期目标是在有限路径代价下最大化地图覆盖、frontier 消除和定位约束质量。
可以写成概念上的多目标优化：

$$
\max_{\pi}
\sum_{t=0}^{T}
\left[
\alpha\,\Delta \mathrm{Cov}(\mathcal{M}_t)
+
\beta\,\Delta D(\mathcal{G}^{P}_t)
-
\gamma\,L(P_t)
-
\eta\,\mathrm{Risk}_t
\right],
$$

其中策略 $\pi$ 把 $\mathcal{S}_t$ 映射为候选动作。不同 `slam_mode` 的差别主要是：

- 如何构造 $\mathcal{A}_t$；
- 如何定义评分函数 $J_m$；
- 是否使用 $\mathcal{G}^{P}_t$ 或 $\mathcal{G}^{T}_t$；
- 何时从宏观探索切换到 frontier 补覆盖或局部清扫。

当前代码没有直接优化上述长期目标，而是用 Nav2 可达性预检加一系列可解释的贪心、
图启发式或 TSP 子问题近似它。

## 3. 各方法的统一描述

### 3.1 `frontier`: Safe-Frontier Baseline

候选集合直接取共享 safe-frontier 池：

$$
\mathcal{A}_t = \mathcal{A}^{F}_t.
$$

方法评分为

$$
J_{\mathrm{frontier}}(a_C)
=
U_F(a_C \mid \mathbf{x}_t).
$$

实现上先按 $U_F$ 排序，只检查前 `frontier_planning_attempts` 个候选。Nav2 返回的路径
只用于可达性验证；一旦找到第一个可达 frontier goal，就发送
`NavigateToPose`：

$$
a_t^\star
=
\mathrm{firstReachable}
\left(
\mathrm{sort}_{\downarrow U_F}
(\mathcal{A}^{F}_t)
\right).
$$

到达后可按 frontier 类型执行 Nav2 `Spin` 和 `DriveOnHeading` probe，以向 unknown
或 open edge 方向继续获取观测。

### 3.2 `approx_graph`: 近似 Pose Graph Scoring

候选仍来自共享 safe-frontier 池：

$$
\mathcal{A}_t = \mathcal{A}^{F}_t.
$$

但方法不会选择第一个可达候选，而是对多个可达路径做 hallucination。对候选 $a$，
将 Nav2 路径 $P_t(\mathbf{q}(a))$ 采样成虚拟节点，并复制当前近似图得到

$$
\widehat{\mathcal{G}}^{P}_t(a)
=
\mathcal{G}^{P}_t
\oplus
\mathrm{Hallucinate}\left(P_t(\mathbf{q}(a)), \mathcal{M}_t\right).
$$

路径附近 unknown ratio 会提高 odometry-like edge information；occupied ratio
超过阈值时会尝试添加几何近似 loop edge。评分为

$$
J_{\mathrm{approx}}(a)
=
D\left(\widehat{\mathcal{G}}^{P}_t(a)\right)
-
\lambda_P\,L\left(P_t(\mathbf{q}(a))\right).
$$

因此

$$
a_t^\star
=
\arg\max_{a \in \mathcal{R}_t(\mathcal{A}^{F}_t)}
J_{\mathrm{approx}}(a).
$$

该方法把“可能改善图约束的路径”作为 frontier 选择依据，但并不修改
`slam_toolbox` 内部后端。

### 3.3 `gbsae`: 先验 Topo-Metric Graph Route

GBSAE 使用 per-world 先验图

$$
\mathcal{G}^{B}=(\mathcal{V}^{B},\mathcal{E}^{B}),
$$

其中边权为欧氏长度 $\ell_e$，信息权重为

$$
w_e = \frac{1}{\ell_e}.
$$

初始顶点为离机器人最近的先验顶点：

$$
v_0
=
\arg\min_{v \in \mathcal{V}^{B}}
\left\|
\mathbf{z}_v-\mathbf{p}(\mathbf{x}_0)
\right\|_2.
$$

基础路线是 nearest-unvisited greedy route，并用先验图最短路展开为连续顶点序列

$$
\tau_B = (v_0,v_1,\ldots,v_K).
$$

路线构造时可插入谱 loop revisit。若候选回环边 $e$ 带来的 spanning-tree D-opt 增益
超过额外往返代价，则插入：

$$
J_{\mathrm{loop}}(e)
=
\Delta D_{\mathrm{tree}}(e)
-
\lambda_L\,2\ell_e.
$$

运行时 active step 为 $v_k$。若 $\mathbf{z}_{v_k}$ 当前是 known-free 且 Nav2 可达，
候选为先验顶点目标：

$$
\mathcal{A}_t = \{\mathbf{z}_{v_k}\}.
$$

若 active vertex 当前不可直接导航，则把 safe-frontier 候选分配给尚未完成的先验顶点：

$$
\mathrm{assign}(a_C)
=
\arg\min_{v \in \mathcal{V}^{B}\setminus \mathcal{V}^{\mathrm{done}}}
\left\|
\mathbf{p}_C-\mathbf{z}_v
\right\|_2.
$$

只尝试分配给 active vertex 的 frontier：

$$
\mathcal{A}_t =
\{a_C \in \mathcal{A}^{F}_t
\mid
\mathrm{assign}(a_C)=v_k
\}.
$$

先验路线结束后，方法回到标准 frontier coverage。

### 3.4 `gvd_gbsae`: Online GVD Bootstrap to GBSAE

该方法先从当前地图和粗世界边界构造 obstacle-only GVD 图：

$$
\mathcal{G}^{G}_t =
\mathrm{Compress}
\left(
\mathrm{Skeletonize}
\left(
\Omega^{\mathrm{trav}}_t
\right)
\right).
$$

其中 obstacle 被膨胀后不可通行，free 与 unknown 在 GVD bootstrap 中都视为可探索：

$$
\Omega^{\mathrm{trav}}_t
=
\Omega^{\mathrm{bounds}}
\setminus
\mathrm{Inflate}(\mathcal{O}_t,r_{\mathrm{robot}}).
$$

GVD bootstrap 的候选来自 skeleton 或压缩拓扑图上的目标点：

$$
\mathcal{A}_t =
\mathcal{A}^{G}_t
=
\{a_v \mid v \in \mathcal{V}^{G}_t\}.
$$

局部 GVD A* 给出到候选的骨架偏置路径，评分综合目标距离、边界未扫掠方向、轨迹重叠惩罚
和当前朝向一致性。可抽象为

$$
J_{\mathrm{gvd}}(a_v)
=
\alpha_U U_{\mathrm{unswept}}(v)
-
\alpha_D d_G(\mathbf{x}_t,v)
-
\alpha_H H_{\mathrm{overlap}}(v)
+
\alpha_A A_{\mathrm{heading}}(v).
$$

实际执行前仍用 Nav2 预检：

$$
a_t^\star
=
\mathrm{firstReachable}
\left(
\mathrm{sort}_{\downarrow J_{\mathrm{gvd}}}
(\mathcal{A}^{G}_t)
\right).
$$

当轨迹扫掠覆盖达到阈值后，当前 live GVD component 被转换为 GBSAE 风格图：

$$
\mathcal{G}^{B}_{\mathrm{live}}
\leftarrow
\mathrm{GBSAEGraph}(\mathcal{G}^{G}_t),
$$

随后执行与 `gbsae` 相同的 route、loop revisit、frontier 分配和最终 frontier 补覆盖。

GVD 图连通性会优先使用 GVD-first connection；必要时使用 clearance 更宽松的双向 A*
作为 switching fallback。fallback edge 只是拓扑连通假设，最终运动仍必须通过 Nav2。

### 3.5 `gvd_hierarchical`: 宏观 GVD 与局部 Frontier 清扫

该方法始终使用 live GVD 图作为宏观骨架：

$$
\mathcal{G}^{T}_t = \mathcal{G}^{G}_t.
$$

状态中维护已探索顶点集合 $\mathcal{V}^{\mathrm{exp}}_t$ 与已清扫顶点集合
$\mathcal{V}^{\mathrm{clr}}_t$。宏观 required target 分为 expansion targets 和
cleanup targets：

$$
\mathcal{V}^{\mathrm{req}}_t
=
\mathcal{V}^{\mathrm{expand}}_t
\cup
\mathcal{V}^{\mathrm{cleanup}}_t.
$$

已清扫顶点不再作为必访目标，但仍可作为最短路 transit。宏观路线由开放 TSP 近似：

$$
\tau_G
=
\mathrm{OpenTSP}
\left(
\mathcal{G}^{G}_t,
v_{\mathrm{start}},
\mathcal{V}^{\mathrm{req}}_t
\right).
$$

下一宏观目标是 $\tau_G$ 中尚未到达的下一个顶点：

$$
a_t^{\mathrm{macro}}
=
v_{\mathrm{next}} \in \tau_G.
$$

当机器人到达一个 cleanup-eligible 的 endpoint 或 degenerate leaf 时，进入该顶点的局部
Region 清扫。Region 是以顶点附近 free/unknown seed 为中心的矩形 flood：

$$
\mathcal{Q}_t(v)
=
\arg\max_{\mathcal{Q}}
\left[
\lambda_A
\frac{\mathrm{area}(\mathcal{Q})}{\mathrm{area}_{\max}}
+
\lambda_S
\mathrm{squareness}(\mathcal{Q})
\right],
$$

并受 occupied cell、粗世界边界和其他 GVD 顶点约束。

局部清扫候选是接触 Region 的 frontier：

$$
\mathcal{A}^{Q}_t(v)
=
\{a_C \in \mathcal{A}^{F}_t \mid C \cap \mathcal{Q}_t(v) \neq \varnothing\}.
$$

基础局部排序使用简单 size-distance 规则：

$$
J_{\mathrm{local}}(a_C)
=
\frac{|C|}
{\left\|\mathbf{g}(C)-\mathbf{p}(\mathbf{x}_t)\right\|_2+\epsilon_d}.
$$

若启用局部 approximate graph，则对 Region 内 Nav2 可达路径使用与
`approx_graph` 相同的 D-opt 评分：

$$
J_{\mathrm{local\_graph}}(a)
=
D\left(\widehat{\mathcal{G}}^{P,Q}_t(a)\right)
-
\lambda_P L(P_t(\mathbf{q}(a))).
$$

局部清扫完成条件为

$$
\rho_t(v) \ge \rho_{\min}
\quad
\text{or}
\quad
\mathcal{R}_t(\mathcal{A}^{Q}_t(v)) = \varnothing.
$$

之后标记 $v \in \mathcal{V}^{\mathrm{clr}}_t$ 并回到宏观 GVD 路线。所有宏观和局部阶段结束后，
方法进入全局 frontier tail cleanup，即再次使用标准 frontier 选择完成剩余覆盖。

## 4. 方法对比

| `slam_mode` | 候选集合 $\mathcal{A}_t$ | 评分 $J_m$ | 图结构 | 结束或切换 |
| --- | --- | --- | --- | --- |
| `frontier` | $\mathcal{A}^{F}_t$ | $I_t(C)/(d+\epsilon_d)$ | 无额外图 | frontier 稳定消失后完成 |
| `approx_graph` | $\mathcal{A}^{F}_t$ | hallucinated D-opt 减路径代价 | $\mathcal{G}^{P}_t$ | frontier 稳定消失后完成 |
| `gbsae` | active prior vertex 或其分配 frontier | prior route 与可选 loop revisit | $\mathcal{G}^{B}$ | route 完成后 frontier 补覆盖 |
| `gvd_gbsae` | GVD skeleton 目标，之后 live GBSAE/frontier | GVD sweep utility，之后 GBSAE | $\mathcal{G}^{G}_t \rightarrow \mathcal{G}^{B}_{\mathrm{live}}$ | sweep 阈值后切 GBSAE |
| `gvd_hierarchical` | 宏观 GVD vertex 或局部 Region frontier | open TSP，局部 size-distance 或 D-opt | $\mathcal{G}^{G}_t$ 与局部 $\mathcal{G}^{P,Q}_t$ | 宏观/局部完成后 tail cleanup |

## 5. 实现边界

上述 formulation 描述的是探索层的问题建模。当前实现的重要边界是：

- `slam_toolbox` 后端未修改，探索层不读取其内部实时 pose graph。
- 所有真正运动都通过 Nav2 action 执行，`exploration_coordinator` 不发布 `/cmd_vel`。
- D-opt 分数是候选目标排序启发式，不等同于后端真实信息矩阵优化。
- GVD 方法把 unknown 当作可探索通行空间来构建骨架，但最终目标仍需 Nav2 在当前代价地图上验证。
- GVD switching fallback edge 表示可能连通性，RViz 中与原始 GVD edge 区分显示，执行前仍由 Nav2 审核。
