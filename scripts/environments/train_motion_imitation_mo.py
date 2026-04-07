import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Anymal-D-v0")
parser.add_argument("--motion", type=str, required=True)
parser.add_argument("--mode", type=str, default="bc",
                    choices=["bc", "rl", "eval", "debug_reward"])
parser.add_argument("--num_envs", type=int, default=64)

# BC
parser.add_argument("--bc_epochs", type=int, default=300)
parser.add_argument("--bc_batch", type=int, default=2048)
parser.add_argument("--bc_lr", type=float, default=3e-4)
parser.add_argument("--bc_wd", type=float, default=1e-4, help="L2 weight decay")
parser.add_argument("--bc_collect_s", type=float, default=60.0)
parser.add_argument("--bc_phase_obs", action="store_true", default=True,
                    help="Add sin/cos phase encoding to observations (recommended)")

# RL
parser.add_argument("--rl_iters", type=int, default=3000)
parser.add_argument("--rl_steps", type=int, default=32)
parser.add_argument("--rl_lr", type=float, default=3e-5)
parser.add_argument("--sigma_pos", type=float, default=0.25,
                    help="Sigma for joint position tracking reward")
parser.add_argument("--sigma_vel", type=float, default=1.0,
                    help="Sigma for joint velocity tracking reward")
parser.add_argument("--w_pos", type=float, default=0.05)
parser.add_argument("--w_vel", type=float, default=0.1)
parser.add_argument("--w_alive", type=float, default=0.2,
                    help="Reward for staying alive and upright")
parser.add_argument("--w_fwd", type=float, default=0.65,
                    help="Forward velocity reward weight")
parser.add_argument("--w_fwd_vel", type=float, default=0.015,
                    help="Small additional reward for forward velocity (helps learning direction)")
parser.add_argument("--w_smooth", type=float, default=0.01,
                    help="Action smoothness reward weight")
parser.add_argument("--w_disp", type=float, default=0.70,
                    help="Net displacement reward weight — rewards sustained forward "
                        "progress over a rolling window, not instantaneous velocity.")
parser.add_argument("--disp_horizon", type=int, default=10,
                    help="Window (in steps) over which to measure displacement. "
                        "10 steps = 0.2s at 50Hz. Backward steps cancel forward ones.")

parser.add_argument("--vf_warmup_iters", type=int, default=100,
                    help="Iters to warm-up value net only when resuming (policy frozen)")
parser.add_argument("--bc_reg_floor", type=float, default=0.10,
                    help="Minimum BC-KL regularization weight (never anneals below this)")

# Shared
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--eval_vel_x", type=float, default=0.5,
                    help="Forward velocity command during eval in m/s")
parser.add_argument("--resume_ckpt", type=str, default=None,
                    help="Resume RL training from an existing RL checkpoint")
parser.add_argument("--bc_ckpt", type=str, default=None)
parser.add_argument("--save_dir", type=str, default="./checkpoints")
parser.add_argument("--stiffness", type=float, default=200.0)
parser.add_argument("--damping", type=float, default=5.0)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.actuators import DCMotorCfg
from tqdm import trange
import logging
from tqdm import tqdm
import hashlib
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# Generate a unique 8-char hash based on the current timestamp
run_hash = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
run_type = args_cli.mode.upper()
args_cli.save_dir = os.path.join(args_cli.save_dir, run_type + "_" + run_hash)

log.info(f"Results will be saved to: {args_cli.save_dir}")

ISAAC_DEFAULT = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.4, -0.4, 0.4, -0.4,
    -0.8, 0.8, -0.8, 0.8,
], dtype=np.float32)

PYB_TO_ISAAC = np.array([0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11], dtype=int)
SIGN_FLIP = np.ones(12, dtype=np.float32)
SIGN_FLIP[2] = -1.0
SIGN_FLIP[3] = -1.0

OBS_DIM = 48
ACTION_DIM = 12
ACTION_SCALE = 0.5
PHASE_DIM = 2  # sin + cos phase encoding


def load_motion(npz_path: str) -> dict:
    data = np.load(npz_path)
    frame_dt = float(data["frame_duration"])
    raw_pyb = np.asarray(data["joint_pos"], dtype=np.float32)
    raw_isaac = raw_pyb[:, PYB_TO_ISAAC]
    clip_mean = raw_isaac.mean(axis=0)
    delta = (raw_isaac - clip_mean) * SIGN_FLIP[None, :]
    joint_pos = delta + ISAAC_DEFAULT[None, :]
    T = raw_pyb.shape[0]
    duration = T * frame_dt

    # Pre-compute joint velocities for velocity reward
    joint_vel = np.diff(joint_pos, axis=0, prepend=joint_pos[:1]) / frame_dt  # (T,12)

    print(f"[Motion] {T} frames  dt={frame_dt:.4f}s  duration={duration:.2f}s")

    # Diagnose smoothness
    steps = np.abs(np.diff(joint_pos, axis=0))
    max_step = steps.max()
    max_vel = max_step / frame_dt
    if max_vel > 5.0:
        print(f"[Motion] WARNING: max instantaneous velocity = {max_vel:.1f} rad/s")
        print(f"         This suggests unsmoothed data. Run smooth_and_diagnose_motion.py first!")
    else:
        print(f"[Motion] Max instantaneous velocity: {max_vel:.2f} rad/s ✓")

    return {
        "joint_pos": joint_pos, "joint_vel": joint_vel,
        "frame_dt": frame_dt, "T": T, "duration": duration,
    }


def get_ref_batch(clip: dict, t_arr: np.ndarray) -> tuple:
    """Get (joint_pos, joint_vel) reference for N envs. t_arr: (N,)"""
    jp = clip["joint_pos"]
    jv = clip["joint_vel"]
    dt = clip["frame_dt"]
    T = clip["T"]
    u = (t_arr / dt) % T
    i0 = np.floor(u).astype(int)
    i1 = (i0 + 1) % T
    a = (u - i0)[:, None]
    pos = (1 - a) * jp[i0] + a * jp[i1]
    vel = (1 - a) * jv[i0] + a * jv[i1]
    return pos.astype(np.float32), vel.astype(np.float32)


def phase_encoding(t_arr: np.ndarray, duration: float) -> np.ndarray:
    """Encode phase as (sin, cos) so policy knows where it is in gait cycle."""
    phase = (t_arr / duration) * 2 * np.pi  # (N,)
    return np.stack([np.sin(phase), np.cos(phase)], axis=-1).astype(np.float32)  # (N,2)


class MLP(nn.Module):
    """
    Policy: [obs (48) + phase (2)] → action (12).
    Phase encoding (sin/cos) is critical — without it the policy cannot
    know where it is in the gait cycle and will average over all phases.
    """

    def __init__(self, obs_dim: int, action_dim: int,
                hidden=(512, 256, 128), stochastic: bool = False,
                use_phase: bool = True):
        super().__init__()
        self.use_phase = use_phase
        in_dim = obs_dim + (PHASE_DIM if use_phase else 0)
        layers = []
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.stochastic = stochastic
        if stochastic:
            self.log_std = nn.Parameter(torch.zeros(action_dim) - 3.5)
            self.log_std.requires_grad_(False)

    def forward(self, obs: torch.Tensor, phase: torch.Tensor = None):
        if self.use_phase and phase is not None:
            x = torch.cat([obs, phase], dim=-1)
        else:
            x = obs
        h = self.net(x)
        mean = self.mean_head(h)
        if self.stochastic:
            std = self.log_std.exp().expand_as(mean)
            return torch.distributions.Normal(mean, std)
        return mean

    def act(self, obs, phase=None, deterministic=False):
        if self.stochastic:
            d = self.forward(obs, phase)
            return d.mean if deterministic else d.rsample()
        return self.forward(obs, phase)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, use_phase: bool = True):
        super().__init__()
        in_dim = obs_dim + (PHASE_DIM if use_phase else 0)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs, phase=None):
        if phase is not None:
            x = torch.cat([obs, phase], dim=-1)
        else:
            x = obs
        return self.net(x).squeeze(-1)


def compute_reward(
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        q_ref: torch.Tensor,
        qd_ref: torch.Tensor,
        obs: torch.Tensor,
        prev_action: torch.Tensor = None,
        curr_action: torch.Tensor = None,
        disp_vel: torch.Tensor = None,
        sigma_pos: float = 0.25,
        sigma_vel: float = 1.0,
        w_pos: float = 0.05,
        w_vel: float = 0.0,
        w_alive: float = 0.2,
        w_disp: float = 0.70,
        w_fwd_vel: float = 0.0,
        w_smooth: float = 0.01,
) -> tuple:
    # Joint position tracking
    dq = joint_pos - q_ref
    pos_rew = torch.exp(-(dq ** 2).sum(-1) / (sigma_pos ** 2))

    dqd = joint_vel - qd_ref
    vel_rew = torch.exp(-(dqd ** 2).sum(-1) / (sigma_vel ** 2))

    # Upright bonus
    gravity_proj = obs[:, 6:9]
    upright = torch.exp(-10.0 * (gravity_proj[:, :2] ** 2).sum(-1))

    # Velocity command
    vel_actual = obs[:, 0]
    vel_cmd = obs[:, 9]
    has_cmd = (vel_cmd.abs() > 0.05).float()

    # disp_vel = (x[t] - x[t-H]) / (H * dt)  - average velocity over the window.
    # Backward phases directly subtract from displacement → no oscillation exploit.
    if disp_vel is not None:
        vel_for_tracking = disp_vel
    else:
        vel_for_tracking = vel_actual  # fallback (e.g., debug mode)

    vel_error = (vel_for_tracking - vel_cmd) ** 2
    disp_track = (has_cmd * torch.exp(-vel_error / 0.25) + (1 - has_cmd) * 0.5)

    # Small instantaneous forward bonus for gradient signal during early steps
    # (before the window fills up). Can be zeroed via w_fwd_vel=0.
    fwd_vel = vel_actual.clamp(0.0, 2.0) / 2.0

    # Action smoothness
    if prev_action is not None and curr_action is not None:
        action_diff = ((curr_action - prev_action) ** 2).mean(-1)
        smooth_rew = torch.exp(-action_diff / 0.04)
    else:
        smooth_rew = torch.ones(joint_pos.shape[0], device=joint_pos.device)

    total = (w_pos * pos_rew
            + w_vel * vel_rew
            + w_alive * upright
            + w_disp * disp_track
            + w_fwd_vel * fwd_vel
            + w_smooth * smooth_rew)

    # Logging: fraction of envs with positive displacement this step
    if disp_vel is not None:
        fwd_ratio = (disp_vel > 0.05).float().mean().item()
        back_ratio = (disp_vel < -0.05).float().mean().item()
        disp_vel_mean = disp_vel.mean().item()
    else:
        fwd_ratio = back_ratio = disp_vel_mean = 0.0

    return total, {
        "pos_rew": pos_rew.mean().item(),
        "vel_rew": vel_rew.mean().item(),
        "disp_track": disp_track.mean().item(),
        "disp_vel": disp_vel_mean,
        "upright": upright.mean().item(),
        "smooth": smooth_rew.mean().item(),
        "fwd_ratio": fwd_ratio,
        "back_ratio": back_ratio,
        "total": total.mean().item(),
    }


def make_env(num_envs):
    env_cfg = parse_env_cfg(task_name=args_cli.task, device="cuda:0", num_envs=num_envs)
    dc_cfg = DCMotorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        saturation_effort=120.0, effort_limit=80.0, velocity_limit=7.5,
        stiffness={".*": args_cli.stiffness},
        damping={".*": args_cli.damping},
    )
    env_cfg.scene.robot = env_cfg.scene.robot.replace(actuators={"legs": dc_cfg})
    return gym.make(args_cli.task, cfg=env_cfg)


def init_to_motion(env, clip, phase_t, device):
    robot = env.unwrapped.scene["robot"]
    num_envs = env.unwrapped.num_envs
    q_np, _ = get_ref_batch(clip, phase_t)
    q_t = torch.tensor(q_np, dtype=torch.float32, device=device)
    v_t = torch.zeros_like(q_t)
    jids = env.unwrapped.action_manager._terms["joint_pos"]._joint_ids
    robot.write_joint_state_to_sim(q_t, v_t, joint_ids=jids)
    env.unwrapped.scene.write_data_to_sim()


def get_obs(obs, key: str = "policy") -> torch.Tensor:
    """
    Extract observation tensor from Isaac Lab's observation dict.
    """
    if isinstance(obs, dict):
        if key in obs:
            return obs[key]
        first = next(iter(obs.values()))
        if isinstance(first, torch.Tensor):
            return first
        raise ValueError(f"Obs dict keys: {list(obs.keys())}")
    if isinstance(obs, torch.Tensor):
        return obs
    raise TypeError(f"Expected dict or Tensor, got {type(obs)}")


def print_obs_info(obs):
    """Print what's in the observation dict — call this once at startup."""
    if isinstance(obs, dict):
        print("[obs_info] Observation dict:")
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                print(f"  '{k}': shape={v.shape}  min={v.min():.4f}  max={v.max():.4f}")
    elif isinstance(obs, torch.Tensor):
        print(f"[obs_info] Observation tensor: shape={obs.shape}")


def collect_bc_data(env, clip, device):
    robot = env.unwrapped.scene["robot"]
    num_envs = env.unwrapped.num_envs
    sim_dt = getattr(env.unwrapped, "step_dt", 0.02)
    default_pos = robot.data.default_joint_pos[0].cpu().numpy()
    n_steps = int(args_cli.bc_collect_s / sim_dt)

    obs_list, act_list, phase_list = [], [], []
    phase_t = np.random.uniform(0, clip["duration"], size=num_envs)
    obs, _ = env.reset()
    print_obs_info(obs)
    init_to_motion(env, clip, phase_t, device)

    print(f"[BC] Collecting {n_steps} steps × {num_envs} envs = "
        f"{n_steps * num_envs:,} samples")

    for step in range(n_steps):
        q_ref_np, _ = get_ref_batch(clip, phase_t)
        action_np = (q_ref_np - default_pos[:12]) / ACTION_SCALE
        acts = torch.tensor(action_np, dtype=torch.float32, device=device)

        obs_cpu = get_obs(obs).cpu()
        obs_list.append(obs_cpu)
        act_list.append(acts.cpu())
        phase_list.append(torch.tensor(
            phase_encoding(phase_t, clip["duration"]), dtype=torch.float32))

        obs, _, terminated, truncated, _ = env.step(acts)
        phase_t += sim_dt
        done = (terminated | truncated).cpu().numpy()
        if done.any():
            new_p = np.random.uniform(0, clip["duration"], size=num_envs)
            phase_t[done] = new_p[done]
            init_to_motion(env, clip, phase_t, device)

        if step % 1000 == 0:
            print(f"  step {step}/{n_steps}  {len(obs_list) * num_envs:,} samples")

    return (torch.cat(obs_list),
            torch.cat(act_list),
            torch.cat(phase_list))


def train_bc(obs_buf, act_buf, phase_buf, policy, device):
    policy = policy.to(device)
    optim = torch.optim.AdamW(policy.parameters(),
                            lr=args_cli.bc_lr, weight_decay=args_cli.bc_wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, args_cli.bc_epochs, eta_min=1e-5)
    N = obs_buf.shape[0]

    obs_d = obs_buf.to(device)
    act_d = act_buf.to(device)
    phase_d = phase_buf.to(device)

    save_path = os.path.join(args_cli.save_dir, "policy_bc.pt")
    best_loss = float("inf")
    train_losses, val_losses = [], []

    # 90/10 train/val split
    split = int(N * 0.9)
    idx = torch.randperm(N, device=device)
    tr_i = idx[:split]
    va_i = idx[split:]

    print(f"\n[BC] {args_cli.bc_epochs} epochs  "
        f"batch={args_cli.bc_batch}  lr={args_cli.bc_lr}  wd={args_cli.bc_wd}")
    print(f"     train={len(tr_i):,}  val={len(va_i):,}")

    for epoch in range(args_cli.bc_epochs):
        policy.train()
        perm = torch.randperm(len(tr_i), device=device)
        ep_loss = 0.0
        n_b = 0
        for start in range(0, len(tr_i), args_cli.bc_batch):
            b = tr_i[perm[start:start + args_cli.bc_batch]]
            pred = policy(obs_d[b], phase_d[b])
            loss = nn.functional.mse_loss(pred, act_d[b])
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            ep_loss += loss.item()
            n_b += 1
        sched.step()
        train_loss = ep_loss / n_b

        # Validation
        policy.eval()
        with torch.no_grad():
            val_pred = policy(obs_d[va_i], phase_d[va_i])
            val_loss = nn.functional.mse_loss(val_pred, act_d[va_i]).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({"epoch": epoch, "val_loss": best_loss,
                        "state_dict": policy.state_dict()}, save_path)

        if epoch % 25 == 0 or epoch == args_cli.bc_epochs - 1:
            gap = val_loss - train_loss
            flag = "  ← OVERFIT" if gap > train_loss * 2 else ""
            print(f"  epoch {epoch:4d}  train={train_loss:.5f}  "
                f"val={val_loss:.5f}  gap={gap:.5f}{flag}  "
                f"best_val={best_loss:.5f}")

    print(f"[BC] Best val loss: {best_loss:.5f}  saved → {save_path}")

    # Plot train/val curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(train_losses, label="train")
        ax.semilogy(val_losses, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss (log)")
        ax.set_title("BC Training — train vs val loss")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(args_cli.save_dir, "bc_loss.png"), dpi=120)
        print(f"[BC] Loss plot → {args_cli.save_dir}/bc_loss.png")
        plt.close()
    except Exception as e:
        print(f"[BC] Plot failed: {e}")

    return policy


class PPOBuffer:
    def __init__(self, num_envs, horizon, obs_dim, action_dim, device):
        self.obs = torch.zeros(horizon, num_envs, obs_dim, device=device)
        self.phase = torch.zeros(horizon, num_envs, PHASE_DIM, device=device)
        self.acts = torch.zeros(horizon, num_envs, action_dim, device=device)
        self.logps = torch.zeros(horizon, num_envs, device=device)
        self.rewards = torch.zeros(horizon, num_envs, device=device)
        self.dones = torch.zeros(horizon, num_envs, device=device)
        self.values = torch.zeros(horizon, num_envs, device=device)
        self.ptr = 0
        self.horizon = horizon

    def store(self, obs, phase, acts, logps, rewards, dones, values):
        t = self.ptr % self.horizon
        self.obs[t] = obs
        self.phase[t] = phase
        self.acts[t] = acts
        self.logps[t] = logps
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        self.ptr += 1

    def compute_returns(self, last_val, gamma=0.99, lam=0.95):
        H = self.horizon
        N = self.obs.shape[1]
        adv = torch.zeros(H, N, device=self.obs.device)
        gae = torch.zeros(N, device=self.obs.device)
        for t in reversed(range(H)):
            nv = last_val if t == H - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * nv * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            adv[t] = gae
        returns = adv + self.values
        return adv.reshape(-1), returns.reshape(-1)

    def flat(self):
        return (self.obs.reshape(-1, self.obs.shape[-1]),
                self.phase.reshape(-1, PHASE_DIM),
                self.acts.reshape(-1, self.acts.shape[-1]),
                self.logps.reshape(-1), self.values.reshape(-1))


def train_ppo(env, clip, policy, value_net, device, is_resume: bool = False):
    import copy
    # Create a frozen copy of the BC policy for regularization
    bc_policy_frozen = copy.deepcopy(policy).to(device)
    for p in bc_policy_frozen.parameters():
        p.requires_grad_(False)
    bc_policy_frozen.eval()
    bc_reg_weight = 1.0

    robot = env.unwrapped.scene["robot"]
    num_envs = env.unwrapped.num_envs
    sim_dt = getattr(env.unwrapped, "step_dt", 0.02)
    default_pos = robot.data.default_joint_pos[0].cpu().numpy()
    default_t = torch.tensor(default_pos[:12], device=device)

    jids = env.unwrapped.action_manager._terms["joint_pos"]._joint_ids

    try:
        cm = env.unwrapped.command_manager
        bv = cm._terms["base_velocity"]
        bv.cfg.ranges.lin_vel_x = (0.4, 0.6)
        bv.cfg.ranges.lin_vel_y = (0.0, 0.0)
        bv.cfg.ranges.ang_vel_z = (0.0, 0.0)
        print("[RL] Velocity commands set to vx=0.4-0.6 m/s")
    except Exception as e:
        print(f"[RL] Could not set velocity commands: {e}")

    policy_optim = torch.optim.Adam(policy.parameters(), lr=args_cli.rl_lr)
    value_optim = torch.optim.Adam(value_net.parameters(), lr=args_cli.rl_lr * 3)

    buf = PPOBuffer(num_envs, args_cli.rl_steps, OBS_DIM, ACTION_DIM, device)
    save_path = os.path.join(args_cli.save_dir, "policy_rl.pt")

    phase_t = np.random.uniform(0, clip["duration"], size=num_envs)
    obs, _ = env.reset()
    init_to_motion(env, clip, phase_t, device)
    ep_return = torch.zeros(num_envs, device=device)
    ep_returns = deque(maxlen=200)
    rew_log = deque(maxlen=200)
    best_return = -float("inf")

    history_iters = []
    history_ret = []
    history_pg = []
    history_vf = []
    history_comps = {}

    print(f"\n[PPO] {args_cli.rl_iters} iters  "
        f"σ_pos={args_cli.sigma_pos}  σ_vel={args_cli.sigma_vel}")
    print(f"      w_pos={args_cli.w_pos}  w_vel={args_cli.w_vel}  "
        f"w_alive={args_cli.w_alive}  w_fwd={args_cli.w_fwd}")
    if is_resume:
        print(f"[PPO] Resuming — value net warmup for {args_cli.vf_warmup_iters} iters "
            f"(policy frozen)")

    # Displacement buffer: tracks x-position for each env over last disp_horizon steps.
    # disp_vel[i] = (x[t] - x[t-H]) / (H * dt)  — backward steps directly cancel forward ones.
    disp_horizon = args_cli.disp_horizon
    pos_x_buf = torch.zeros(num_envs, disp_horizon + 1, device=device)
    pos_x_buf_ready = torch.zeros(num_envs, dtype=torch.bool, device=device)

    prev_acts = torch.zeros(num_envs, ACTION_DIM, device=device)

    for it in trange(args_cli.rl_iters, desc="PPO Training"):
        if it == 200 and policy.stochastic:
            policy.log_std.requires_grad_(True)
            tqdm.write("\n[RL] Unfreezing log_std at iter 200")

        in_vf_warmup = is_resume and (it < args_cli.vf_warmup_iters)
        if in_vf_warmup and it == 0:
            for p in policy.parameters():
                p.requires_grad_(False)
            tqdm.write("[PPO] Value-net warmup: policy parameters frozen")
        elif in_vf_warmup and it == args_cli.vf_warmup_iters - 1:
            for p in policy.parameters():
                p.requires_grad_(True)
            # Re-apply the log_std freeze so the iter-200 unfreeze still applies
            if policy.stochastic and it < 200:
                policy.log_std.requires_grad_(False)
            tqdm.write(f"[PPO] Value-net warmup complete at iter {it} — policy unfrozen")

        policy.eval()
        value_net.eval()

        with torch.no_grad():
            for _ in range(args_cli.rl_steps):
                obs_t = get_obs(obs).to(device)
                if it == 0 and _ == 0:  # only first iter, first step
                    print(f"[DEBUG] obs[:,0] stats: mean={obs_t[:, 0].mean():.3f}  "
                        f"std={obs_t[:, 0].std():.3f}  min={obs_t[:, 0].min():.3f}  max={obs_t[:, 0].max():.3f}")
                    print(f"[DEBUG] obs[:,9] stats: mean={obs_t[:, 9].mean():.3f}  "
                        f"std={obs_t[:, 9].std():.3f}  (should be ~0.5)")
                    print(f"[DEBUG] obs[0,:] = {obs_t[0].cpu().numpy().round(3)}")
                phase_np = phase_encoding(phase_t, clip["duration"])
                phase_t_ = torch.tensor(phase_np, dtype=torch.float32, device=device)

                dist = policy(obs_t, phase_t_)
                acts = dist.rsample()
                logps = dist.log_prob(acts).sum(-1)
                values = value_net(obs_t, phase_t_)

                next_obs, _, terminated, truncated, _ = env.step(acts)
                done = (terminated | truncated).float()
                phase_t += sim_dt

                # Reward computation
                q_cur = robot.data.joint_pos[:, :12].clone()
                qd_cur = robot.data.joint_vel[:, :12].clone()
                q_ref_np, qd_ref_np = get_ref_batch(clip, phase_t)
                q_ref = torch.tensor(q_ref_np, dtype=torch.float32, device=device)
                qd_ref = torch.tensor(qd_ref_np, dtype=torch.float32, device=device)
                # Update displacement buffer (circular shift, newest x in last slot)
                cur_x = robot.data.root_state_w[:, 0]  # base x position, world frame
                pos_x_buf = torch.roll(pos_x_buf, -1, dims=1)
                pos_x_buf[:, -1] = cur_x
                pos_x_buf_ready |= True  # after first step, all envs have at least 1 sample

                # disp_vel in m/s: net forward progress over the window
                disp_vel = (pos_x_buf[:, -1] - pos_x_buf[:, 0]) / (disp_horizon * sim_dt)

                reward, comps = compute_reward(
                    q_cur, qd_cur, q_ref, qd_ref, obs_t,
                    prev_action=prev_acts,
                    curr_action=acts,
                    disp_vel=disp_vel,
                    sigma_pos=args_cli.sigma_pos,
                    sigma_vel=args_cli.sigma_vel,
                    w_pos=args_cli.w_pos,
                    w_vel=args_cli.w_vel,
                    w_alive=args_cli.w_alive,
                    w_disp=args_cli.w_disp,
                    w_fwd_vel=0.0,
                    w_smooth=args_cli.w_smooth)
                prev_acts = acts.clone()
                rew_log.append(comps)

                buf.store(obs_t, phase_t_, acts, logps, reward, done, values)
                ep_return += reward

                reset_mask = done.bool().cpu().numpy()
                if reset_mask.any():
                    new_p = np.random.uniform(0, clip["duration"], size=num_envs)
                    phase_t[reset_mask] = new_p[reset_mask]
                    init_to_motion(env, clip, phase_t, device)
                    prev_acts[done.bool()] = 0.0
                    reset_idx = done.bool()
                    if reset_idx.any():
                        pos_x_buf[reset_idx] = robot.data.root_state_w[reset_idx, 0:1]
                    for r in ep_return[done.bool()]:
                        ep_returns.append(r.item())
                    ep_return[done.bool()] = 0.0

                obs = next_obs

            last_val = value_net(get_obs(obs).to(device),
                                torch.tensor(phase_encoding(phase_t, clip["duration"]),
                                            dtype=torch.float32, device=device))
            adv, ret = buf.compute_returns(last_val)

        # PPO update
        policy.train()
        value_net.train()
        obs_f, ph_f, act_f, olp_f, _ = buf.flat()
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg_l = vf_l = ent_l = 0.0
        n_up = 0

        for _ in range(4):
            idx = torch.randperm(obs_f.shape[0], device=device)
            for s in range(0, idx.shape[0], 512):
                b = idx[s:s + 512]
                dist = policy(obs_f[b], ph_f[b])
                nlp = dist.log_prob(act_f[b]).sum(-1)
                ratio = (nlp - olp_f[b]).exp()
                pg = -torch.min(ratio * adv_n[b],
                                ratio.clamp(0.8, 1.2) * adv_n[b]).mean()
                vf = nn.functional.mse_loss(value_net(obs_f[b], ph_f[b]), ret[b])
                ent = dist.entropy().sum(-1).mean()

                with torch.no_grad():
                    bc_dist = bc_policy_frozen(obs_f[b], ph_f[b])

                bc_kl = torch.distributions.kl_divergence(dist, bc_dist).sum(-1).mean()

                bc_w = bc_reg_weight * max(args_cli.bc_reg_floor, 1.0 - it / 1000.0)

                # During value-net warmup, skip the policy gradient term entirely.
                if in_vf_warmup:
                    loss = vf  # only update value net
                else:
                    loss = pg + 0.5 * vf - 0.01 * ent + bc_w * bc_kl

                value_optim.zero_grad()
                if not in_vf_warmup:
                    policy_optim.zero_grad()

                loss.backward()

                nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
                value_optim.step()

                if not in_vf_warmup:
                    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    policy_optim.step()

                pg_l += pg.item() if not in_vf_warmup else 0.0
                vf_l += vf.item()
                ent_l += ent.item() if not in_vf_warmup else 0.0
                n_up += 1

        mean_ret = np.mean(ep_returns) if ep_returns else 0.0
        if mean_ret > best_return and ep_returns:
            best_return = mean_ret
            torch.save({"iter": it, "return": best_return,
                        "policy": policy.state_dict(),
                        "value": value_net.state_dict()}, save_path)

        if it % 100 == 0 or it == args_cli.rl_iters - 1:
            # Average reward components
            if rew_log:
                avg = {k: np.mean([r[k] for r in rew_log]) for k in rew_log[0]}
                comp_str = "  ".join(f"{k}={v:.3f}" for k, v in avg.items())

                history_iters.append(it)
                history_ret.append(mean_ret)
                history_pg.append(pg_l / n_up)
                history_vf.append(vf_l / n_up)
                for k, v in avg.items():
                    if k not in history_comps:
                        history_comps[k] = []
                    history_comps[k].append(v)

            else:
                comp_str = ""
            warmup_flag = " [VF_WARMUP]" if in_vf_warmup else ""
            bc_w_cur = bc_reg_weight * max(args_cli.bc_reg_floor, 1.0 - it / 1000.0)
            fwd_ratio = np.mean([1.0 if r.get("fwd_vel", 0) > 0.05 else 0.0 for r in rew_log])
            back_ratio = np.mean([1.0 if r.get("back_pen", 0) > 0.05 else 0.0 for r in rew_log])
            tqdm.write(
                f"iter {it:5d}{warmup_flag}  ret={mean_ret:.3f}  best={best_return:.3f}  "
                f"pg={pg_l / n_up:.4f}  vf={vf_l / n_up:.4f}  bc_w={bc_w_cur:.3f} | {comp_str} "
                f"fwd_ratio={fwd_ratio:.2f}  back_ratio={back_ratio:.2f}"
            )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        # Mean Return Plot
        axs[0].plot(history_iters, history_ret, label="Mean Return", color="blue")
        axs[0].set_title("PPO: Mean Return")
        axs[0].set_ylabel("Return")
        axs[0].grid(True)
        axs[0].legend()

        # Losses Plot
        axs[1].plot(history_iters, history_pg, label="PG Loss", color="red")
        axs[1].plot(history_iters, history_vf, label="VF Loss", color="green")
        axs[1].set_title("PPO: Losses")
        axs[1].set_ylabel("Loss")
        axs[1].grid(True)
        axs[1].legend()

        # Reward Components Plot
        for k, v in history_comps.items():
            axs[2].plot(history_iters, v, label=k)
        axs[2].set_title("PPO: Reward Components")
        axs[2].set_xlabel("Iteration")
        axs[2].set_ylabel("Value")
        axs[2].grid(True)
        axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        fig.tight_layout()
        plot_path = os.path.join(args_cli.save_dir, "ppo_metrics.png")
        fig.savefig(plot_path, dpi=120, bbox_inches='tight')
        print(f"\n[PPO] Training metrics plot saved → {plot_path}")
        plt.close()
    except Exception as e:
        print(f"\n[PPO] Plotting failed: {e}")

    try:
        import csv
        csv_path = os.path.join(args_cli.save_dir, "ppo_metrics.csv")
        with open(csv_path, mode="w", newline="") as f:
            # Dynamically build the header
            comp_keys = list(history_comps.keys())
            header = ["iteration", "mean_return", "pg_loss", "vf_loss"] + comp_keys
            writer = csv.writer(f)
            writer.writerow(header)

            # Write the data rows
            for i in range(len(history_iters)):
                row = [
                    history_iters[i],
                    history_ret[i],
                    history_pg[i],
                    history_vf[i]
                ]
                # Add the reward components for this iteration
                for k in comp_keys:
                    row.append(history_comps[k][i])
                writer.writerow(row)

        print(f"[PPO] Training metrics CSV saved → {csv_path}")
    except Exception as e:
        print(f"[PPO] CSV export failed: {e}")


def debug_reward(env, clip, device):
    """
    Run reference motion for a few seconds and print reward components
    at each step.  Helps verify the reward function is working before training.
    """
    robot = env.unwrapped.scene["robot"]
    sim_dt = getattr(env.unwrapped, "step_dt", 0.02)
    default_pos = robot.data.default_joint_pos[0].cpu().numpy()
    num_envs = env.unwrapped.num_envs

    phase_t = np.zeros(num_envs)
    obs, _ = env.reset()
    init_to_motion(env, clip, phase_t, device)

    print("\n[Debug] Running reference motion and logging reward components...")
    print(f"  All rewards should be in (0,1]. "
        f"pos_rew≈0.9+ is good. upright≈1.0 means stable.")
    print(f"\n  {'step':>5}  {'pos_rew':>8}  {'vel_rew':>8}  {'vel_track':>9}  "
        f"{'upright':>8}  {'back_pen':>9}  {'fwd_vel':>8}  {'total':>8}")

    for step in range(200):
        q_ref_np, qd_ref_np = get_ref_batch(clip, phase_t)
        action_np = (q_ref_np - default_pos[:12]) / ACTION_SCALE
        acts = torch.tensor(action_np, dtype=torch.float32, device=device)

        obs, _, terminated, truncated, _ = env.step(acts)
        phase_t += sim_dt

        q_cur = robot.data.joint_pos[:, :12]
        qd_cur = robot.data.joint_vel[:, :12]
        q_ref = torch.tensor(q_ref_np, dtype=torch.float32, device=device)
        qd_ref = torch.tensor(qd_ref_np, dtype=torch.float32, device=device)
        obs_t = get_obs(obs).to(device)

        _, comps = compute_reward(
            q_cur, qd_cur, q_ref, qd_ref, obs_t,
            prev_action=None, curr_action=None,
            sigma_pos=args_cli.sigma_pos,
            sigma_vel=args_cli.sigma_vel,
            w_pos=args_cli.w_pos,
            w_vel=args_cli.w_vel,
            w_alive=args_cli.w_alive,
            w_fwd=args_cli.w_fwd,
            w_fwd_vel=args_cli.w_fwd_vel)

        if step < 20 or step % 20 == 0:
            print(f"  {step:5d}  {comps['pos_rew']:8.4f}  {comps['vel_rew']:8.4f}  "
                f"{comps['vel_track']:9.4f}  {comps['upright']:8.4f}  "
                f"{comps['fwd_vel']:8.4f}  {comps['back_pen']:9.4f}  {comps['total']:8.4f}")

        if (terminated | truncated).any():
            print(f"  [RESET at step {step}]")
            phase_t = np.zeros(num_envs)
            obs, _ = env.reset()
            init_to_motion(env, clip, phase_t, device)

    print("\n[Debug] Expected values with smoothed reference motion:")
    print("  pos_rew ≈ 0.85-0.95  (some lag from PD controller is normal)")
    print("  vel_rew ≈ 0.70-0.90  (joint velocity tracking)")
    print("  vel_track ≈ 0.50 (neutral when no cmd) or 0.90+ when tracking commanded velocity")
    print("  upright ≈ 0.95-1.00  (robot stays upright)")
    print("  fwd_vel ≈ 0.00-0.05  (near zero — reference has no base velocity)")
    print("  total   ≈ 0.75-0.90")
    print("\n  If pos_rew < 0.5: motion data is too noisy or sigma_pos is too tight")
    print("  If upright < 0.8: robot is falling — check stiffness or initial pose")


def main():
    device = "cuda:0"
    clip = load_motion(args_cli.motion)
    n_envs = 1 if args_cli.mode == "eval" else args_cli.num_envs
    env = make_env(n_envs)

    if args_cli.mode == "bc":
        os.makedirs(args_cli.save_dir, exist_ok=True)
        policy = MLP(OBS_DIM, ACTION_DIM, stochastic=False,
                    use_phase=args_cli.bc_phase_obs)
        obs_b, act_b, ph_b = collect_bc_data(env, clip, device)
        policy = train_bc(obs_b, act_b, ph_b, policy, device)

    elif args_cli.mode == "rl":
        os.makedirs(args_cli.save_dir, exist_ok=True)
        policy = MLP(OBS_DIM, ACTION_DIM, stochastic=True, use_phase=True)
        value_net = ValueNet(OBS_DIM, use_phase=True)
        is_resume = False

        # Resume from RL checkpoint (has both policy + value net)
        if args_cli.resume_ckpt and os.path.exists(args_cli.resume_ckpt):
            ck = torch.load(args_cli.resume_ckpt, map_location=device, weights_only=False)
            policy.load_state_dict(ck["policy"])
            if "value" in ck:
                value_net.load_state_dict(ck["value"])
            is_resume = True
            print(f"[RL] Resumed from RL checkpoint: {args_cli.resume_ckpt}")
            print(f"[RL] Value-net warmup enabled for {args_cli.vf_warmup_iters} iters")
        else:
            # Init from BC checkpoint
            init_ckpt = args_cli.bc_ckpt or args_cli.ckpt
            if init_ckpt and os.path.exists(init_ckpt):
                ck = torch.load(init_ckpt, map_location=device, weights_only=False)
                sd = ck.get("state_dict") or ck.get("policy")
                # Load with strict=False since stochastic adds log_std
                policy.load_state_dict(sd, strict=False)
                print(f"[RL] Loaded init from {init_ckpt}")

        train_ppo(env, clip, policy.to(device), value_net.to(device), device,
                is_resume=is_resume)

    elif args_cli.mode == "debug_reward":
        policy = None  # don't need policy for reward debug
        debug_reward(env, clip, device)

    elif args_cli.mode == "eval":
        ck = torch.load(args_cli.ckpt, map_location=device, weights_only=False)
        is_rl = "policy" in ck
        policy = MLP(OBS_DIM, ACTION_DIM, stochastic=is_rl, use_phase=True)
        sd = ck.get("policy") or ck.get("state_dict")
        policy.load_state_dict(sd)
        policy = policy.to(device).eval()

        robot = env.unwrapped.scene["robot"]
        sim_dt = getattr(env.unwrapped, "step_dt", 0.02)
        phase_t = np.random.uniform(0, clip["duration"], size=n_envs)
        obs, _ = env.reset()
        init_to_motion(env, clip, phase_t, device)
        default_pos = robot.data.default_joint_pos[0].cpu().numpy()

        # Set velocity commands in eval too — same as training
        try:
            cm = env.unwrapped.command_manager
            bv = cm._terms["base_velocity"]
            bv.cfg.ranges.lin_vel_x = (args_cli.eval_vel_x, args_cli.eval_vel_x)
            bv.cfg.ranges.lin_vel_y = (0.0, 0.0)
            bv.cfg.ranges.ang_vel_z = (0.0, 0.0)
            # Force immediate resample so obs[9] gets the new command NOW
            cm.reset(torch.arange(n_envs, device=device))
            print(f"[Eval] Velocity commands set to vx={args_cli.eval_vel_x} m/s")
        except Exception as e:
            print(f"[Eval] Could not set velocity commands: {e}")

        print(f"\n[Eval] Running policy ({args_cli.ckpt})")
        step = 0
        ep_step = 0
        ep_count = 0
        vel_history = []
        while simulation_app.is_running():
            with torch.no_grad():
                obs_t = get_obs(obs).to(device)
                ph_t = torch.tensor(
                    phase_encoding(phase_t, clip["duration"]),
                    dtype=torch.float32, device=device)
                acts = policy.act(obs_t, ph_t, deterministic=True)

            obs, _, terminated, truncated, _ = env.step(acts)
            phase_t += sim_dt
            ep_step += 1

            # Print every step for first 3 episodes, then every 50 steps
            if ep_count < 3 or step % 50 == 0:
                obs_cpu = get_obs(obs).cpu()
                vel_history.append(obs_cpu[0, 0].item())
                if len(vel_history) > 20:
                    vel_history.pop(0)
                rolling_mean = sum(vel_history) / len(vel_history)
                print(f"[Eval] ep={ep_count} step={ep_step:4d}  "
                    f"vel_cmd={obs_cpu[0, 9]:.3f}  vel_actual={obs_cpu[0, 0]:.3f}  "
                    f"upright_z={obs_cpu[0, 8]:.3f}  "
                    f"fall={'YES' if (terminated | truncated).any() else 'no'}  "
                    f"roll_mean_vel={rolling_mean:.3f}")

            if (terminated | truncated).any():
                print(f"[Eval] >>> Episode {ep_count} ended at step {ep_step} "
                    f"({ep_step * sim_dt:.2f}s)")
                ep_count += 1
                ep_step = 0
                phase_t = np.zeros(n_envs)
                obs, _ = env.reset()
                init_to_motion(env, clip, phase_t, device)
            step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
