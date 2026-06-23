# Exploration Coordinator State Machine

This document summarizes the runtime state machine implemented by
`src/activeslam/activeslam/exploration_coordinator.py`.

The coordinator has one explicit `state` field, plus three mode-specific
substate fields that refine behavior while `state == SELECTING` or
`state == NAVIGATING`:

- `selection_kind`: `frontier`, `gbsae_vertex`, `gbsae_frontier`, `gvd_goal`,
  `gvd_obstruction_replan`, `hierarchical_gvd_vertex`, or
  `hierarchical_local_frontier`.
- `current_navigation_kind`: `frontier`, `gbsae_vertex`, `gvd_goal`, or
  `hierarchical_gvd_vertex`.
- `gvd_phase`: `bootstrap`, `gbsae`, `macro`, `local_clear`, or
  `tail_cleanup`, depending on `slam_mode`.

## Top-level state chart

```mermaid
stateDiagram-v2
    [*] --> WAITING_FOR_NAV2: __init__

    WAITING_FOR_NAV2 --> WAITING_FOR_NAV2: Nav2 not ready or no map yet
    WAITING_FOR_NAV2 --> INITIAL_SPIN: Nav2 ready and latest_map exists

    INITIAL_SPIN --> IDLE: initial Spin callback\nsuccess or failure

    IDLE --> SELECTING: retry delay elapsed\npose available
    IDLE --> COMPLETE: standard selection sees\nno frontiers and stable map
    IDLE --> RANDOM_RECOVERY_SPIN: GVD progress watchdog expired

    SELECTING --> SELECTING: next candidate path request
    SELECTING --> NAVIGATING: Nav2 path found\nand goal dispatched
    SELECTING --> IDLE: path timeout, no candidate,\nor retry scheduled
    SELECTING --> RANDOM_RECOVERY_SPIN: GVD progress watchdog expired

    NAVIGATING --> IDLE: Nav2 done, timed out,\nor failure handled
    NAVIGATING --> ALIGNING_FRONTIER_PROBE: frontier goal reached\nand probe needs yaw alignment
    NAVIGATING --> PROBING_FRONTIER: frontier goal reached\nand already aligned
    NAVIGATING --> SELECTING: active GVD path obstructed\nstart obstruction replan
    NAVIGATING --> RANDOM_RECOVERY_SPIN: GVD progress watchdog expired

    ALIGNING_FRONTIER_PROBE --> PROBING_FRONTIER: Spin succeeded
    ALIGNING_FRONTIER_PROBE --> IDLE: Spin failed or timed out

    PROBING_FRONTIER --> IDLE: DriveOnHeading succeeded,\nfailed, or timed out

    RANDOM_RECOVERY_SPIN --> RANDOM_RECOVERY_DRIVE: random Spin succeeded
    RANDOM_RECOVERY_SPIN --> RANDOM_RECOVERY_SPIN: Spin failed and\nattempts remain
    RANDOM_RECOVERY_SPIN --> IDLE: attempts exhausted

    RANDOM_RECOVERY_DRIVE --> IDLE: random drive succeeded
    RANDOM_RECOVERY_DRIVE --> RANDOM_RECOVERY_SPIN: drive failed and\nattempts remain
    RANDOM_RECOVERY_DRIVE --> IDLE: attempts exhausted

    COMPLETE --> COMPLETE: terminal
```

## Selection dispatch by mode

```mermaid
flowchart TD
    A[IDLE calls _start_selection] --> B{slam_mode}

    B -->|frontier| S[_start_standard_selection]
    B -->|approx_graph| S

    B -->|gbsae| G{GBSAE planner ready?}
    G -->|yes| GS[_start_gbsae_selection]
    G -->|no| S

    B -->|gvd_gbsae| GG{gvd_phase}
    GG -->|bootstrap and sweep below switch ratio| GVD[_start_gvd_selection]
    GG -->|bootstrap and switch succeeds| GS
    GG -->|gbsae and planner ready| GS
    GG -->|otherwise| S

    B -->|gvd_hierarchical| H{gvd_phase}
    H -->|macro| HM[_start_hierarchical_macro_selection]
    H -->|local_clear| HL[_start_hierarchical_local_selection]
    H -->|tail_cleanup| S

    S --> C{frontiers?}
    C -->|no, map stable| COMPLETE
    C -->|no, map not stable| IDLE
    C -->|yes| SELECTING

    GS --> GV{active GBSAE step?}
    GV -->|known-free vertex| SELECTING
    GV -->|vertex unavailable| GF[_start_gbsae_frontier_selection]
    GV -->|route complete| S
    GF --> SELECTING

    GVD --> SELECTING
    HM --> SELECTING
    HM -->|final macro vertex ready| HL
    HM -->|macro route complete| S
    HL --> SELECTING
    HL -->|local candidates exhausted| IDLE
```

## SELECTING callbacks

```mermaid
flowchart TD
    SELECTING --> K{selection_kind}

    K -->|frontier| F[_path_computed]
    F -->|path found, frontier mode| NAV[NAVIGATING frontier]
    F -->|path found, approx_graph| FS[score and cache graph candidate]
    FS --> FNext[next frontier candidate]
    F -->|no path| FNext
    FNext -->|candidates remain| F
    FNext -->|graph candidate exists| NAV
    FNext -->|none reachable| IDLE

    K -->|gbsae_vertex| GV[_gbsae_vertex_path_computed]
    GV -->|path found| GVN[NAVIGATING gbsae_vertex]
    GV -->|optional revisit unreachable| IDLE
    GV -->|required vertex unreachable| GF[_start_gbsae_frontier_selection]

    K -->|gbsae_frontier| GFF[_gbsae_frontier_path_computed]
    GFF -->|path found| NAV
    GFF -->|no path, candidates remain| GFF
    GFF -->|none reachable, advance route| IDLE

    K -->|gvd_goal| GVD[_gvd_path_computed]
    GVD -->|path found| GVDN[NAVIGATING gvd_goal]
    GVD -->|no path, candidates remain| GVD
    GVD -->|none reachable| IDLE

    K -->|gvd_obstruction_replan| OR[_gvd_obstruction_replan_computed]
    OR -->|checkpoint reachable| GVDN
    OR -->|checkpoint unreachable, more remain| OR
    OR -->|none reachable| IDLE

    K -->|hierarchical_gvd_vertex| HG[_hierarchical_gvd_path_computed]
    HG -->|path found| HGN[NAVIGATING hierarchical_gvd_vertex]
    HG -->|no path| IDLE

    K -->|hierarchical_local_frontier| HLF[_hierarchical_local_path_computed]
    HLF -->|path found, immediate mode| NAV
    HLF -->|path found, approx graph enabled| HScore[score and cache local graph candidate]
    HScore --> HNext[next local candidate]
    HLF -->|no path| HNext
    HNext -->|candidates remain| HLF
    HNext -->|best local graph candidate exists| NAV
    HNext -->|none reachable| IDLE
```

## NAVIGATING completion behavior

```mermaid
flowchart TD
    N[NAVIGATING _navigation_finished] --> K{current_navigation_kind}

    K -->|frontier| F{status succeeded?}
    F -->|yes, probe enabled| P[_start_frontier_probe]
    F -->|yes, no probe| IDLE
    F -->|no| IDLE

    K -->|gbsae_vertex| G{status succeeded?}
    G -->|yes| GA[advance GBSAE step]
    GA --> IDLE
    G -->|no| GF[mark failed or skip optional loop revisit]
    GF --> IDLE

    K -->|gvd_goal| GV{status succeeded?}
    GV -->|yes| IDLE
    GV -->|no, mark failed| IDLE

    K -->|hierarchical_gvd_vertex| H{status succeeded?}
    H -->|yes| HM[mark macro vertex reached]
    HM --> HP{should clear local?}
    HP -->|yes| LC[gvd_phase = local_clear]
    HP -->|no| MC[gvd_phase = macro]
    LC --> IDLE
    MC --> IDLE
    H -->|no| HR[mark failed and route dirty]
    HR --> IDLE
```

## Frontier probe sequence

```mermaid
stateDiagram-v2
    NAVIGATING --> ALIGNING_FRONTIER_PROBE: frontier reached\nprobe enabled\nalignment needed
    NAVIGATING --> PROBING_FRONTIER: frontier reached\nprobe enabled\nalready aligned
    ALIGNING_FRONTIER_PROBE --> PROBING_FRONTIER: Nav2 Spin succeeded
    ALIGNING_FRONTIER_PROBE --> IDLE: Spin failed or timed out
    PROBING_FRONTIER --> IDLE: DriveOnHeading succeeded
    PROBING_FRONTIER --> IDLE: DriveOnHeading failed or timed out
```

## GVD recovery and obstruction replan

```mermaid
flowchart TD
    A[IDLE, SELECTING, or NAVIGATING] --> W{GVD progress watchdog expired?}
    W -->|no| A
    W -->|yes| R[RANDOM_RECOVERY_SPIN]

    R --> RS{Spin succeeded?}
    RS -->|yes| D[RANDOM_RECOVERY_DRIVE]
    RS -->|no, attempts remain| R
    RS -->|no, attempts exhausted| IDLE

    D --> DS{DriveOnHeading succeeded?}
    DS -->|yes| IDLE
    DS -->|no, attempts remain| R
    DS -->|no, attempts exhausted| IDLE

    N[NAVIGATING gvd_goal] --> O{active GVD path obstructed?}
    O -->|yes| S[SELECTING gvd_obstruction_replan]
    S --> C{checkpoint path reachable?}
    C -->|yes| N2[NAVIGATING gvd_goal fallback]
    C -->|no, more checkpoints| S
    C -->|no checkpoints left| IDLE
```

## Code anchors

- Explicit state constants are defined in `ExplorationCoordinator` near lines 98-107.
- `_control_loop()` owns the top-level dispatch, timeout checks, watchdog checks,
  and calls into selection from `IDLE`.
- `_start_selection()` dispatches to the mode-specific selection routines.
- `_schedule_retry()` is the common transition back to `IDLE`.
- `_navigation_finished()` handles `NAVIGATING` completion based on
  `current_navigation_kind`.
- `_start_frontier_probe()`, `_frontier_probe_spin_finished()`, and
  `_frontier_probe_drive_finished()` own the frontier probe substates.
- `_gvd_bootstrap_stuck()` and the `_gvd_random_recovery_*()` methods own the
  bounded GVD random recovery substates.
- `_handle_gvd_path_obstructed()` and `_gvd_obstruction_replan_computed()` own
  the GVD obstruction replan path through `SELECTING`.
