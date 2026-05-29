# scripts/environments — Training Scripts

This directory contains the policy training scripts for motion imitation on ANYmal-D. The three `train_motion_imitation*.py` scripts are variants of each other with separate feature sets. `train_motion_imitation_mo.py` is from Mohamad. Use `train_motion_imitation_ghost.py` for new experiments unless you have a specific reason to use an earlier version.

---

## Script Evolution

```
train_motion_imitation.py          ← baseline BC + PPO, cubic spline interpolation
        │
        ▼  adds: phase encoding, multi-component reward, displacement tracking, BC-KL reg
train_motion_imitation_mo.py       ← "multi-objective" variant
        │
        ▼  adds: ghost robot visualization
train_motion_imitation_ghost.py    ← current main script
```

All three share the same BC → PPO two-stage training flow and the same checkpoint format.

---

## File Overview

The following files are a part of the training suite of code.

```
environments/
├── train_motion_imitation.py       # baseline
├── train_motion_imitation_mo.py    # adds phase obs, displacement reward, KL regularisation
├── train_motion_imitation_ghost.py # adds ghost robot (reference pose visualisation in sim)
├── anymal_motion_playback.py       # visual playbacks
├── motion_to_isaac.py              # standalone coordinate-conversion utility
├── graphing_joint_motion.py        # plot joint trajectories
├── data/                           # .npz motion clipping utilities
│   ├── smooth_and_diagnose_motion.py  # Butterworth + Savitzky-Golay smoother + diagnostics
│   ├── convert_txt_to_npz.py          # convert ANYmal-format JSON motion → .npz
│   └── motions/                       # pre-processed motion arrays
└── checkpoints/                    # saved policy weights (written at runtime)
```

---

## Training Modes

Every script accepts a `--mode` flag:

| Mode | What it does |
|---|---|
| `bc` | Collects rollouts using the reference motion as the oracle policy, then trains a deterministic MLP by supervised regression on `(obs, action)` pairs. Saves `policy_bc.pt`. |
| `rl` | Loads a BC checkpoint, then fine-tunes with PPO using a motion + velocity-tracking reward. Saves `policy_rl.pt`. |
| `eval` | Loads any checkpoint, runs it in the simulator indefinitely, and prints per-step diagnostics. Uses a single env. |
| `debug_reward` | Plays back the reference motion open-loop and prints reward component values at each step. Useful for verifying reward scales before committing to a full training run. Not available in the base script. |

---

## Script Comparison

### `train_motion_imitation.py` — Baseline

The original training script. Keeps things simple.

**Key properties:**
- Motion is evaluated via a **periodic cubic spline** (`scipy.interpolate.CubicSpline`), giving analytically smooth positions and exact first-derivative velocities. This removes jitter from discrete frame-to-frame lookups and was the first fix for staircase artifacts in early training.
- Reward is a single exponential joint-tracking term: `exp(-||q - q_ref||² / σ²)` plus an optional forward velocity component.
- No phase encoding — the observation does not include gait-cycle position, so the policy must implicitly track phase from joint state alone.
- No BC-KL regularization during PPO.

**When to use:** Quick experiments where you don't need the full reward suite or ghost visualization. Also useful as a reference implementation.

```bash
python scripts/environments/train_motion_imitation.py \
    --task   Isaac-Velocity-Flat-Anymal-D-v0 \
    --motion scripts/environments/data/pace_smoothed.npz \
    --mode   bc \
    --num_envs 64 --bc_epochs 200 --bc_collect_s 60.0 \
    --save_dir scripts/environments/checkpoints
```

```bash
python scripts/environments/train_motion_imitation.py \
    --task   Isaac-Velocity-Flat-Anymal-D-v0 \
    --motion scripts/environments/data/pace_smoothed.npz \
    --mode   rl \
    --rl_iters 2000 --rl_sigma 0.25 --rl_vel_weight 0.5 \
    --bc_ckpt scripts/environments/checkpoints/policy_bc.pt \
    --save_dir scripts/environments/checkpoints
```

---

### `train_motion_imitation_mo.py` — Multi-Objective Variant

Extends the baseline with a richer reward and more stable RL training. 

> [!NOTE]
> Note for Mohamad: This is the file that you provided us to help us out when converting to Isaac.

**Added over baseline:**

- **Phase encoding** — trigonometric gait-cycle position appended to every observation. Without this, the policy cannot distinguish between phases of the gait cycle and learns to average over them, producing a shuffling gate.
- **Displacement-based velocity reward** — instead of rewarding instantaneous forward velocity (which can be gamed by oscillating in place), the reward measures net displacement over a rolling window of `--disp_horizon` steps: `disp_vel = (x[t] - x[t-H]) / (H·dt)`. Backward steps directly cancel forward progress, so the policy must commit to sustained forward motion to earn reward.
- **BC-KL regularisation** — a KL penalty toward the frozen BC policy is added to the PPO loss. This prevents catastrophic forgetting of the walking gait that BC learned. The weight anneals from 1.0 to `--bc_reg_floor` over the first 1000 iterations so the policy retains style early and gains freedom later.
- **Value-net warmup on resume** — when loading from an existing RL checkpoint (`--resume_ckpt`), the policy is frozen for `--vf_warmup_iters` iterations while the value net adapts to the current reward scale. This prevents stale advantage estimates from corrupting early policy updates.
- **Action smoothness reward** — penalises large step changes between consecutive actions to reduce mechanical wear and produce more natural motion.
- **`debug_reward` mode** — plays back the reference motion and prints reward component values step-by-step, so you can sanity-check reward scales before training.

**Reward weights (all exposed as CLI flags):**

| Flag | Default | Purpose |
|---|---|---|
| `--w_pos` | 0.05 | Joint position tracking |
| `--w_vel` | 0.1 | Joint velocity tracking |
| `--w_alive` | 0.2 | Upright / not-fallen bonus |
| `--w_disp` | 0.70 | Displacement-based velocity tracking |
| `--w_fwd_vel` | 0.015 | Small instantaneous forward bonus (early gradient signal) |
| `--w_smooth` | 0.01 | Action smoothness |

> [!NOTE]
> `w_pos` is intentionally kept low (0.05) so the robot is free to deviate from the reference gait when the velocity reward demands it. Motion imitation is a soft style constraint here, not the primary objective.

```bash
# BC stage
python scripts/environments/train_motion_imitation_mo.py \
    --task   Isaac-Velocity-Flat-Anymal-D-v0 \
    --motion scripts/environments/data/pace_smoothed.npz \
    --mode   bc --num_envs 64 \
    --bc_epochs 300 --bc_batch 2048 \
    --save_dir scripts/environments/checkpoints

# RL stage
python scripts/environments/train_motion_imitation_mo.py \
    --task   Isaac-Velocity-Flat-Anymal-D-v0 \
    --motion scripts/environments/data/pace_smoothed.npz \
    --mode   rl --num_envs 64 \
    --rl_iters 3000 --w_disp 0.70 --disp_horizon 10 \
    --bc_ckpt scripts/environments/checkpoints/policy_bc.pt \
    --save_dir scripts/environments/checkpoints
```

---

### `train_motion_imitation_ghost.py` — Ghost Robot Variant

Extends `_mo` with a **ghost robot** that floats above the real robot during simulation, displaying the reference motion pose side-by-side for visual comparison.

**Added over `_mo`:**

- **Ghost robot visualization** — a second kinematic body (`ghost_robot`, defined in `AnymalDImitationFlatEnvCfg`) is written to sim every step via `update_shadow_robot()`. It mirrors the real robot's base XY position but floats 1.5 m above it, showing the reference joint configuration at the current phase. This makes it easy to see the deviation between policy and reference without stopping simulation.
- The ghost has gravity and collisions disabled and its actuators zeroed in the env config — it is purely visual.

**When to use:** Whenever you want to visually inspect how well the policy is tracking the reference motion during or after training. The ghost does not affect the reward or observations.

```bash
# BC stage
python scripts/environments/train_motion_imitation_ghost.py \
    --task   Isaac-Velocity-Flat-Anymal-D-v0 \
    --motion scripts/environments/data/pace_smoothed.npz \
    --mode   bc --num_envs 64 \
    --bc_epochs 300 --bc_batch 2048 \
    --save_dir scripts/environments/checkpoints

# RL stage (warm-start from BC)
python scripts/environments/train_motion_imitation_ghost.py \
    --task   Isaac-Velocity-Flat-Anymal-D-v0 \
    --motion scripts/environments/data/pace_smoothed.npz \
    --mode   rl --num_envs 64 \
    --rl_iters 3000 --w_disp 0.70 \
    --bc_ckpt scripts/environments/checkpoints/BC_<hash>/policy_bc.pt \
    --save_dir scripts/environments/checkpoints

# Resume an existing RL run (value-net warmup enabled automatically)
python scripts/environments/train_motion_imitation_ghost.py \
    --motion scripts/environments/data/pace_smoothed.npz \
    --mode   rl --num_envs 64 \
    --resume_ckpt scripts/environments/checkpoints/RL_<hash>/policy_rl.pt \
    --save_dir    scripts/environments/checkpoints

# Evaluate
python scripts/environments/train_motion_imitation_ghost.py \
    --motion scripts/environments/data/pace_smoothed.npz \
    --mode   eval \
    --ckpt   scripts/environments/checkpoints/RL_<hash>/policy_rl.pt

# Check reward scales before training
python scripts/environments/train_motion_imitation_ghost.py \
    --motion scripts/environments/data/pace_smoothed.npz \
    --mode   debug_reward
```

---

## Checkpoint Format

Each run creates a timestamped subdirectory under `--save_dir` (e.g. `BC_3f8a1c2d/`).

| File | Written by | Contents |
|---|---|---|
| `policy_bc.pt` | BC stage | `{"epoch", "val_loss", "state_dict"}` |
| `policy_rl.pt` | RL stage | `{"iter", "return", "policy", "value"}` |
| `bc_loss.png` | BC stage | Train vs. val MSE loss curves |
| `ppo_metrics.png` | RL stage | Return, losses, and reward component plots |
| `ppo_metrics.csv` | RL stage | Same data as CSV for external analysis |

The eval mode auto-detects checkpoint type by checking for the `"policy"` key — RL checkpoints have both `policy` and `value`, BC checkpoints have `state_dict`.

---

## Shared Design Decisions

**Two-stage BC → PPO:** BC produces a deterministic warm-start that already walks roughly correctly. Without it, PPO starts from random actions and rarely discovers stable locomotion within a practical iteration budget.

**Motion as a constraint, not the objective:** In the `_mo` and `_ghost` variants, `w_pos` (joint tracking) is set low (0.05) while `w_disp` (forward progress) is dominant (0.70). The motion clip provides style and gait timing; the RL reward shapes direction and speed.

**Phase encoding:** The gait cycle wraps continuously, so the phase is encoded as `(sin(2πt/T), cos(2πt/T))` rather than a raw fraction. This avoids a discontinuity at the loop boundary that would otherwise appear as a sharp input jump to the network.

**Joint ordering and sign convention:** Motion data from the retargeting pipeline (`retarget.py`) uses PyBullet joint ordering. `PYB_TO_ISAAC` and `SIGN_FLIP` are applied once at load time so all downstream reward and BC computations operate on aligned joints.
