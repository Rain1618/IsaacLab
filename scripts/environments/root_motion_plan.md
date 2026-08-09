# Plan: Diagnosing and plotting average root motion in `train_motion_imitation_plus.py`

Reference for implementation. See `paper.tex` Section "Check 3: Training Without
Root Motion" and Discussion — this file operationalizes that section against
the actual training script.

## Analysis: why root motion cannot appear (three independent causes)

### 1. The reference has no root motion by construction

`scripts/retargeting/retarget.py:304` hardcodes:

```python
out_root_pos = np.tile(np.array([[0.0, 0.0, 0.5]]), (T, 1))
out_root_rot = np.tile(identity, ...)
```

Confirmed in `output_squared.npz`: `root_pos` has exactly zero range on all
three axes across 643 frames. `load_motion()` in
`train_motion_imitation_plus.py:112` also never reads `root_pos`/`root_rot`
even for clips that do carry them (`pace_smoothed.npz`,
`anymal_motion_for_isaaclab.npz`).

For those older clips the recovered root is unusable anyway: net displacement
is 0.011 m (x) / 0.048 m (y) over 16.6 s, against a 0.377 m per-stride
oscillation in y — the gait-locked bias is ~8x the signal riding on it. This
is the quantitative version of the paper's `\tbd{0.035}` / `\tbd{0.03}m`
placeholders in Check 3.

### 2. The scale tuning is anisotropic across three unrelated measurements

`scripts/retargeting/helpers.py:213` (`compute_scale_factors`) computes three
independent ratios — `sx` from stance length, `sy` from stance width, `sz`
from base clearance — applied per-axis at `retarget.py:202`. Stored values in
`output_squared.npz`: `(1.295e-3, 3.016e-3, 9.943e-4)`, i.e. `sy/sx = 2.33`,
`sy/sz = 3.03`. A rigid body under a correct isotropic lift needs one scale;
three different ones shear the body before IK runs at all. This is Eq. (14)'s
kappa made concrete and explains the knee-vs-abduction asymmetry in Check 2.

### 3. The displacement reward measures a quantity that cannot mean progress

- `train_motion_imitation_plus.py:692` reads
  `robot.data.root_state_w[:, 0]` — world-frame x, including the per-env grid
  origin offset.
- `imitation_env_cfg.py` (via `velocity_env_cfg.py` `EventCfg.reset_base`)
  randomizes reset yaw over `(-pi, pi)` every episode.
- `CommandsCfg.base_velocity` has `heading_command=True`,
  `rel_heading_envs=1.0`, `ranges.heading=(-pi, pi)`. Setting
  `bv.cfg.ranges.ang_vel_z = (0.0, 0.0)` at line 600 does nothing: with
  `heading_command` on, `ang_vel_z` is recomputed every step from heading
  error (`velocity_command.py:150`), overriding that range.

Consequence: every env faces a uniformly random heading and is steered toward
a random target heading, while `disp_vel` only credits +world-x. A robot
walking perfectly at 0.5 m/s facing -x scores `disp_vel = -0.5` and is
punished. Averaged over envs, `E[world-x displacement] ~= 0` regardless of
whether robots walk — the term carrying `w_disp=0.70` (the largest single
weight) is near-zero-mean noise under the current heading configuration.

Compounding bugs in the same code path:
- `vel_error = (disp_vel - vel_cmd)**2` compares world-frame displacement
  against a body-frame command (`obs[:, 9]`).
- `disp_vel` depends on a 10-step window that never enters `obs`, so
  `V_phi(s_t, phase_t)` cannot represent it — the advantage estimate for the
  dominant reward term is structurally biased.
- `pos_x_buf` is initialized to zeros while world-x starts at the env-grid
  origin, so the first `disp_vel` computed after reset is
  `x_origin / (disp_horizon * sim_dt)` (tens of m/s for far-out envs) —
  poisons early steps of every episode.
- `debug_reward()` calls `compute_reward(..., w_fwd=args_cli.w_fwd, ...)`
  but `compute_reward` has no `w_fwd` parameter -> `TypeError` on entry to
  `--mode debug_reward`.
- `train_ppo`'s logging line reads `r.get("fwd_vel")` / `r.get("back_pen")`,
  but `compute_reward` returns `fwd_ratio` / `back_ratio` — both printed
  values are always 0.00.
- `args_cli.w_fwd` (default 0.65) is parsed but never used in the RL path.
- `update_shadow_robot` checks `"shadow_robot" in env.unwrapped.scene.keys()`
  but the scene registers `ghost_robot` (see `imitation_env_cfg.py`) — dead
  branch, never executes.
- `pos_x_buf_ready` is written, never read.

**Net effect:** the objective is self-contradictory even before the bugs —
`w_pos=0.05` tracks a reference whose maximizer is stepping in place, against
`w_disp=0.70` demanding 0.4-0.6 m/s of travel that the reward cannot actually
observe correctly. The tracker below exists to make this visible rather than
inferred.

## Implementation plan

### Step 0 — Quantities (per env, per control step)

| Quantity | Definition | Why |
|---|---|---|
| `p` | `root_pos_w - env_origins` | strips the grid offset the reward ignores |
| `yaw0` | `robot.data.heading_w` at last reset | yaw is randomized; world axes aren't meaningful across resets |
| `d_fwd`, `d_lat` | `(p - p0)` rotated by `-yaw0` | the only frame in which "forward" exists |
| `arc_len` | running `sum(||delta_p_xy||)` | path length (denominator for travel efficiency) |
| `cmd_disp` | running `sum(v_cmd_x * dt)` | what was commanded |
| `d_worldx` | `p_x - p0_x` | the metric the current reward actually uses — kept for comparison |

Headline scalar: `travel_ratio = |d_fwd| / arc_len`. ~1 means net progress
matches path length (walks somewhere); ~0 means the legs move but the body
doesn't (stepping in place). This states Check 3 as one number instead of an
inferred displacement.

### Step 1 — `RootMotionTracker` class

Insert after `PPOBuffer`. Pure-torch per-env state, no host sync in the hot
loop except appending finished episodes to a `deque`.

Methods: `reset(env_ids, xy_world, yaw, worldx)`,
`step(xy_world, worldx, vel_cmd_x, dt, done_mask) -> (d_fwd, d_lat, d_worldx, done_ids)`,
`summary() -> dict | None`.

### Step 2 — Wire into `train_ppo`

- Instantiate beside `pos_x_buf`; cache `env.unwrapped.scene.env_origins[:, :2]` once.
- Initial `reset()` right after `init_to_motion` at the top of `train_ppo`.
- `step()` call immediately after the existing `cur_x = ...` read inside the
  rollout loop — reuse that same tensor read, no new sync.
- In the reset block: call `tracker.step(...)` (which internally pops
  completed episodes) using the *post-reset* pose, since `reset_base`
  teleports and re-yaws inside `env.step`. Assert
  `||xy_local[done]|| < 1.0` right after (reset pose range is +-0.5 m) as a
  smoke test that the read is on the correct side of the reset.

### Step 3 — Logging

Add `history_root = {}` (kept separate from `history_comps` — this is a
diagnostic, not a reward component) at the existing `it % 100` checkpoint.
Write `root_motion.csv` alongside `ppo_metrics.csv`.

### Step 4 — The figure (`fig_root_motion.{png,pdf}`)

Same `try/except` pattern as the existing `ppo_metrics.png` block so a
plotting failure never kills a run. Three stacked panels:

- (a) Mean per-episode forward displacement (`disp_fwd_mean`) vs iteration,
  +-1 std across finished episodes. Reference lines: commanded displacement
  (`eval_vel_x * episode_length_s`) dashed, and 0 (reference root motion)
  solid — the gap between them is the result.
- (b) `travel_ratio_mean` vs iteration, reference band at 1.0.
- (c) `disp_worldx_mean` (+-1 std) vs `disp_fwd_mean` (+-1 std) — shows the
  reward's own metric is high-variance/near-zero-mean while the
  heading-aligned metric is flat, i.e. the reward term injects variance
  rather than signal.

### Step 5 — Eval-mode trajectory trace (not yet implemented)

In `--mode eval`, add a top-down XY plot: one line per episode, each rotated
into its own reset-heading frame, reference shown as a point at the origin.

### Step 6 — Control conditions (not yet implemented, requires retarget.py changes)

| | Config | Purpose |
|---|---|---|
| A | stock `Isaac-Velocity-Flat-Anymal-D-v0` reward | upper bound; proves tracker can show real travel |
| B | current script as-is | the failure |
| C | B + synthetic root motion in the reference | paper's configuration (C) |
| D | B + isotropic scale (single `s`, not `sx/sy/sz`) | separates scale problem from root problem |

### Step 7 — Prerequisites for a meaningful figure (config only, not code bugs)

1. For any run used to generate the figure: disable heading randomization
   (`bv.cfg.heading_command = False`, `pose_range["yaw"] = (0.0, 0.0)`).
   Without this, forward displacement in a fixed world frame averages to
   ~0 by symmetry regardless of what the policy learned, and the figure
   becomes uninterpretable. The tracker's heading-aligned frame is a partial
   mitigation but a fixed-heading run is still needed to isolate the effect
   cleanly.
2. Any fix to `disp_vel` itself (env-origin subtraction, seeding
   `pos_x_buf` with current x instead of zero) changes the reward and should
   be treated as a distinct condition (`B'`), not folded silently into `B`,
   so the failure documented by `B` stays reproducible.

## Status

- [x] Step 1: `RootMotionTracker` class
- [x] Step 2: wire into `train_ppo`
- [x] Step 3: logging + CSV
- [x] Step 4: `fig_root_motion` plot
- [ ] Step 5: eval-mode trajectory trace
- [ ] Step 6: control condition runs (A/B/C/D)
- [ ] Step 7: prerequisite config changes for a clean figure

### Implementation note

`ManagerBasedRLEnv.step()` auto-resets terminated envs internally
(`_reset_idx`) before returning, so a pose read taken right after
`env.step()` is already post-teleport for any env that just finished an
episode — the same trap the pre-existing `disp_vel` reward term falls into
for `pos_x_buf`. `RootMotionTracker.step()` handles this by only
accumulating `arc_len`/`cmd_disp`/`prev_xy` for envs that stayed alive that
step, and computing a finished episode's displacement from the last
pre-reset pose (`prev_xy`/`prev_worldx`, saved on the prior call) rather than
the contaminated post-reset read. Verified with a standalone unit test
(rotation math for three headings, episode-boundary correctness, and
re-anchoring) run under `env_isaaclab`'s Python, independent of Isaac Sim.
