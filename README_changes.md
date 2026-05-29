# Motion Imitation for ANYmal-D

This document describes the research additions on top of the upstream [IsaacLab](https://github.com/isaac-sim/IsaacLab) fork. The goal is quadruped motion imitation: capture real animal movement from video, retarget it to an ANYmal-D robot, and train a policy that reproduces that motion in simulation using Behavioral Cloning (BC). This repository covers retargeting and policy training. 

---

## Overview of the Pipeline

```
Animal Video Keypoint CSV
        │
        ▼
[1] scripts/retargeting/preprocess_csv.py   ← CSV → .npz (z-up, smoothed)
        │
        ▼
[2] scripts/retargeting/retarget.py         ← IK retargeting to ANYmal-D joint angles
        │
        ▼  (store .npz files)
[3] scripts/environments/data/              ← .npz motion clips live here
        │
        ▼  (optional smoothing / diagnostics)
[3b] scripts/environments/data/smooth_and_diagnose_motion.py
        │
        ▼ 
[4] scripts/environments/train_motion_imitation.py
        ├── mode=bc   → Behavioral Cloning (supervised, fast)
        ├── mode=rl   → PPO fine-tuning (warm-started from BC)
        └── mode=eval → evaluate a saved checkpoint
        │
        ▼
[5] scripts/environments/checkpoints/       ← saved policies
```

> [!NOTE]
> To obtain the keypoint CSV, you want to utilize our [copydog](https://github.com/Rain1618/copydog) tools repository to obtain the a csv file of reference trajectories. 
> 
> This IsaacLab fork offers scripts starting from the retargeting step of the pipeline only.

---

## Directory Layout

The default IssacLab repository files are generally ignored for readability.

```
scripts/
├── environments/
│   ├── anymal_motion_playback.py           # simple open-loop playback of a motion clip
│   ├── motion_to_isaac.py                  # standalone coordinate-conversion utility
│   ├── graphing_joint_motion.py            # visualise joint trajectories
│   ├── train_motion_imitation.py           # main BC + PPO training entry-point
│   ├── train_motion_imitation_mo.py        # experimental multi-objective variant
│   ├── train_motion_imitation_ghost.py     # experimental extended variant with ghost on-top
│   ├── data/
│   │   ├── convert_txt_to_npz.py           # convert ANYmal motion JSON → .npz
│   │   ├── smooth_and_diagnose_motion.py   # smooth + diagnoise a reference motion clip
│   │   ├── pace.json / pace.npz / ...      # example pace-gait clips
│   │   └── motions/                        # pre-processed motion arrays
│   └── checkpoints/                        # saved policy weights (BC & RL)
│       ├── policy_bc.pt
│       ├── policy_rl.pt
│       └── bc_loss_curve.png
├── retargeting/
│   ├── preprocess_csv.py                   # gets the csv -> retarget-ready .npz
│   ├── retarget.py                         # IK-based retargeting in IsaacLab sim
│   ├── helpers.py                          #  utility classes for retarget.py
│   ├── test_preprocess.py                  # unit tests for preprocess_csv.py
│   ├── data/                               # sample MoCap CSVs and outputs
│   └── viz/plot_3d_skeleton.py             # 3-D skeleton visualiser (modified from `copydog` repository)
└── reinforcement_learning/skrl/train.py    # skrl-based PPO trainer (upstream + minor edits)

source/
├── isaaclab/isaaclab/envs/mdp/terminations.py           # added custom terminations
└── isaaclab_tasks/.../locomotion/velocity/
    ├── imitation_env_cfg.py                             # ImitationRoughEnvCfg base config
    └── config/anymal_d/
        ├── imitation_env_cfg.py                         # AnymalD-specific imitation configs
        ├── agents/skrl_imitation_ppo_cfg.yaml           # SKRL PPO hyperparameters
        └── agents/target_data/dog1_pace.txt             # raw target data reference
```

---

## Step-by-Step Quickstart

### Prerequisites

Follow the standard IsaacLab installation. The motion imitation code additionally requires `scipy`:

```bash
pip install scipy
```

```bash
# Start the IsaacLab environment (activates the conda/venv and sets ISAACLAB_PATH)
./isaaclab.sh -p python.sh
```

---

### Step 1 — Prepare a Motion Clip

**Option A: from an ANYmal-format JSON (e.g. from another research paper or library)**

```bash
python scripts/environments/data/convert_txt_to_npz.py \
    --input  scripts/environments/data/pace.json \
    --output scripts/environments/data/pace.npz
```

The JSON format is expected to have `"Frames"` (shape `[T, 19]`: root_pos(3), root_rpy(4), joint_pos(12)) and `"FrameDuration"`.

**Option B: from a DeepLabCut / MoCap CSV of real animal video**

```bash
# 1. Convert CSV to retarget-ready .npz (coordinate transform + smoothing)
python scripts/retargeting/preprocess_csv.py \
    --csv    scripts/retargeting/data/dog1_slow_with_depth_npz.csv \
    --output scripts/retargeting/data/output.npz \
    --fps    30 \
    --hip-lambda 5.0   # L2 regularisation on thigh keypoints (0 to disable)

# 2. Run IK retargeting inside IsaacLab to get ANYmal-D joint angles
python scripts/retargeting/retarget.py \
    --ref-motion scripts/retargeting/data/output.npz \
    --output     scripts/environments/data/pace.npz \
    --fps        30 \
    --robot      anymal_d \
    [--visualise]          # opens a viewer to watch the IK solve
    [--plot_skeleton]      # shows a live 3-D skeleton plot
```

Smoothing is performed in `preprocess_csv.py` to reduce the overhead from running and adjust `retarget.py`. `retarget.py` is the main script that performs the retargeting.

The retargeter pins the robot base in the air and solves per-leg IK iteratively (damped least squares). It exports a `.npz` containing `joint_pos` (in training format), `frame_duration`, `root_pos`, `joint_vel`, and more.

**Option C: diagnose and smooth an existing clip**

```bash
python scripts/environments/data/smooth_and_diagnose_motion.py \
    --motion   scripts/environments/data/pace.npz \
    --cutoff_hz 3.0 \      # Butterworth low-pass cutoff (Hz)
    --savgol_win 11 \      # Savitzky-Golay window
    --upsample  2          # optional 2× frame-rate upsampling via cubic spline
```

This prints per-joint diagnostics (staircase detection, lag-1 autocorrelation), saves `*_smoothed.npz`, and writes three diagnostic plots alongside the file:
- `motion_smoothing_comparison.png` — raw vs. smoothed position
- `velocity_comparison.png` — raw vs. smoothed velocity
- `psd_analysis.png` — power spectral density

---

### Step 2 — Verify Playback (optional)

Open-loop playback lets you visually confirm the motion data looks correct before training:

```bash
python scripts/environments/anymal_motion_playback.py \
    --task     Isaac-Velocity-Flat-Anymal-D-v0 \
    --motion   scripts/environments/data/pace.npz \
    --num_envs 1
```

---

### Step 3 — Behavioral Cloning

BC bootstraps a deterministic MLP policy by supervised regression on `(observation, reference_action)` pairs collected by rolling out the motion clip.

```bash
python scripts/environments/train_motion_imitation.py \
    --task         Isaac-Velocity-Flat-Anymal-D-v0 \
    --motion       scripts/environments/data/pace_smoothed.npz \
    --mode         bc \
    --num_envs     64 \
    --bc_epochs    200 \
    --bc_batch     1024 \
    --bc_lr        3e-4 \
    --bc_collect_s 60.0 \       # seconds of motion per env to collect
    --save_dir     scripts/environments/checkpoints
```

Outputs:
- `checkpoints/policy_bc.pt` — best checkpoint (lowest MSE)
- `checkpoints/bc_loss_curve.png` — training loss over epochs

**Policy architecture** — MLP `[512, 256, 128]` with ELU activations; input: 48-dim observation, output: 12-dim joint position residuals.

**Observation space** (48-dim, drawn from `Isaac-Velocity-Flat-Anymal-D-v0`):
- base linear velocity (3)
- base angular velocity (3)
- projected gravity (3)
- velocity command (4)
- joint positions relative to default (12)
- joint velocities relative to default (12)
- last action (12)

---

### Step 4 — PPO Fine-Tuning

Fine-tunes from the BC checkpoint using a stochastic MLP policy + value network, with a motion-tracking reward.

```bash
python scripts/environments/train_motion_imitation.py \
    --task       Isaac-Velocity-Flat-Anymal-D-v0 \
    --motion     scripts/environments/data/pace_smoothed.npz \
    --mode       rl \
    --num_envs   64 \
    --rl_iters   2000 \
    --rl_steps   24 \          # PPO horizon (steps per rollout)
    --rl_lr      1e-4 \
    --rl_sigma   0.25 \        # Gaussian width for joint-tracking reward
    --bc_ckpt    scripts/environments/checkpoints/policy_bc.pt \
    --save_dir   scripts/environments/checkpoints
```

**Reward** — exponential joint-tracking reward:

```
r = exp( -||q - q_ref||² / σ² )
```

where `q` is the current joint position vector and `q_ref` is the reference from the cubic-spline interpolated motion clip. Optional velocity terms are available but set to weight 0 by default.

Some reference weights for these reward terms can be found @ [Learning Agile Robotic Locomotion Skills by Imitating Animals](https://github.com/erwincoumans/motion_imitation).

**Motion interpolation** — joint targets are evaluated from a periodic cubic spline fitted to the motion frames. This gives smooth positions and analytically correct velocities and helps prevents motion jittering during BC.

Outputs:
- `checkpoints/policy_rl.pt` — best checkpoint (highest mean episode return)

---

### Step 5 — Evaluate a Checkpoint

```bash
python scripts/environments/train_motion_imitation.py \
    --task   Isaac-Velocity-Flat-Anymal-D-v0 \
    --motion scripts/environments/data/pace_smoothed.npz \
    --mode   eval \
    --ckpt   scripts/environments/checkpoints/policy_rl.pt \
    --num_envs 1
```

The script auto-detects whether the checkpoint is from BC (deterministic) or RL (stochastic) and uses deterministic inference for evaluation.

---

## Coordinate Conventions

Two coordinate frames appear throughout this codebase:

| Frame | Up axis | Used by |
|---|---|---|
| **MoCap source** (DeepLabCut) | +Y | raw CSV output |
| **IsaacLab** | +Z | everything else |

Conversion: `(x, y, z)_iso = (x_src, -z_src, y_src)` — a proper rotation of +90° about X (determinant +1, handedness preserved). Applied once in `preprocess_csv.py`.

**Joint ordering** also differs between the ANYmal-D PyBullet training convention and the IsaacLab articulation order. The permutation `PYB_TO_ISAAC = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]` and sign flips for joints 2 and 3 are applied in `train_motion_imitation.py` when loading motion clips. We utilize this new ordering for training.

---

## Environment Configurations

Two new ManagerBasedRL environment configs are provided for the imitation task:

**`ImitationRoughEnvCfg`** ([source/isaaclab_tasks/.../imitation_env_cfg.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/imitation_env_cfg.py))
- Subclasses the standard rough-terrain locomotion env
- Adds a `ghost_robot` slot in the scene (for future reference-motion visualisation — a kinematic ghost that tracks the motion clip)
- Retains standard velocity-tracking rewards and curriculum

**`AnymalDImitationRoughEnvCfg`** ([config/anymal_d/imitation_env_cfg.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/imitation_env_cfg.py))
- Binds `ANYMAL_D_CFG` to both `robot` and `ghost_robot`
- Ghost robot: gravity disabled, collisions disabled, actuator stiffness/damping zeroed

Flat-terrain and play variants are also defined (`AnymalDImitationFlatEnvCfg`, `*_PLAY`).

---

## Key Files Reference

| File | Purpose |
|---|---|
| [scripts/retargeting/preprocess_csv.py](scripts/retargeting/preprocess_csv.py) | MoCap CSV → `.npz`; coordinate transform, confidence interpolation, hip L2 regularisation, two-pass root estimation |
| [scripts/retargeting/helpers.py](scripts/retargeting/helpers.py) | `QuadrupedRetargeter` (DLS IK per leg), scale factor computation, quaternion math, `load_reference_motion` |
| [scripts/retargeting/retarget.py](scripts/retargeting/retarget.py) | Runs the retargeter inside a live IsaacLab sim, exports `.npz` with all kinematics |
| [scripts/environments/data/convert_txt_to_npz.py](scripts/environments/data/convert_txt_to_npz.py) | Convert ANYmal-format JSON motion clip to `.npz` |
| [scripts/environments/data/smooth_and_diagnose_motion.py](scripts/environments/data/smooth_and_diagnose_motion.py) | Cubic spline + Savitzky-Golay + Butterworth pipeline; staircase / autocorrelation diagnostics |
| [scripts/environments/train_motion_imitation.py](scripts/environments/train_motion_imitation.py) | BC data collection + training, PPO rollout + update, evaluation |
| [scripts/environments/anymal_motion_playback.py](scripts/environments/anymal_motion_playback.py) | Open-loop motion playback for visual debugging |
| [source/.../imitation_env_cfg.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/imitation_env_cfg.py) | Base imitation RL env config with ghost robot |
| [source/.../anymal_d/imitation_env_cfg.py](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/imitation_env_cfg.py) | ANYmal-D-specific imitation configs |

---

## Saved Checkpoints

> [!WARNING]
> The checkpoints yield similar results to what we've seen with the Apr 21st presentation. These checkpoints definitely do not work as of yet.

Pre-trained checkpoints are included for reference. All were trained on the pace-gait clip (`pace_smoothed.npz`):

| File | Type | Notes |
|---|---|---|
| `checkpoints/policy_bc.pt` | BC | Latest BC run |
| `checkpoints/policy_bc_feb_27.pt` | BC | Feb 27 snapshot |
| `checkpoints/policy_bc_feb_28_good.pt` | BC | Best BC result |
| `checkpoints/policy_rl.pt` | RL (PPO) | PPO fine-tuned from BC |
| `checkpoints/BC_bd175a8d/policy_bc.pt` | BC | Archived run |
| `checkpoints/RL_d61211e0/policy_rl.pt` | RL | Archived run |

Checkpoint format:
- **BC**: `{"epoch": int, "loss": float, "state_dict": OrderedDict}`
- **RL**: `{"iter": int, "return": float, "policy": OrderedDict, "value": OrderedDict}`

---

## Known Limitations / Open TODOs

- Root state initialisation during RL resets re-randomises joint phases but does not reset the base pose — only joint states are written to sim.
- The ghost robot infrastructure in the env config is not yet wired up in the training loop (no motion-tracking visualisation during training). Use `train_motion_imitation_ghost.py` for ghost infra.
- The motion-tracking reward currently only uses joint position error. Joint velocity and forward velocity reward terms exist in code but have their weights set to 0. This is modified in the other versions of `train_motion_imitation`.
- `preprocess_csv.py` assumes a straight-walk prior (constant heading) when estimating root orientation; it will not accurately handle turning sequences. **This one of the largest limitations we've been working on throughout April**. It seems like the DeepLabCut clip produces a skew bias on the heading angle due to occlusions. We require a methodology that infers this accurate before *behaviour cloning*.
- `retarget.py` currently requires `--num_envs 1`.
- Jittery motions caused my occlusions are resolved by convoluting using an averaging filter / fitting cubic splines. There are other methods we've tried such as using a Kalman Filter (reference [here](https://github.com/Rain1618/copydog)), but there needs to be a more sound way of figuring this out. A potential solution is to train a model that can interpolate this time series well. 