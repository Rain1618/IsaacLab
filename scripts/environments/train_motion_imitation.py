import argparse
from isaaclab.app import AppLauncher

import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Anymal-D-v0")
parser.add_argument("--motion", type=str, required=True)
parser.add_argument("--mode", type=str, default="bc", choices=["bc", "rl", "eval"],
                    help="bc=behavioral cloning, rl=PPO motion tracking, eval=evaluate checkpoint")
parser.add_argument("--num_envs", type=int, default=64,
                    help="Parallel envs for data collection / RL training. Use 1 for eval.")

# BC args
parser.add_argument("--bc_epochs", type=int, default=200)
parser.add_argument("--bc_batch", type=int, default=1024)
parser.add_argument("--bc_lr", type=float, default=3e-4)
parser.add_argument("--bc_collect_s", type=float, default=60.0,
                    help="Seconds of motion to collect per env for BC dataset.")

# RL args
parser.add_argument("--rl_iters", type=int, default=2000)
parser.add_argument("--rl_steps", type=int, default=24,
                    help="Rollout steps per PPO iteration (horizon).")
parser.add_argument("--rl_lr", type=float, default=1e-4)
parser.add_argument("--rl_sigma", type=float, default=0.25,
                    help="Sigma for joint-tracking reward: exp(-||dq||^2 / sigma^2).")
parser.add_argument("--rl_vel_weight", type=float, default=0.5,
                    help="Weight for forward velocity reward component.")

# Shared
parser.add_argument("--ckpt", type=str, default=None,
                    help="Checkpoint to load for eval, or BC checkpoint to init RL from.")
parser.add_argument("--bc_ckpt", type=str, default=None,
                    help="BC checkpoint to use as RL initialization.")
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
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from collections import deque
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.actuators import ActuatorNetLSTMCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab.assets.asset_base import AssetBase

os.makedirs(args_cli.save_dir, exist_ok=True)

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
    print(f"[Motion] {T} frames  dt={frame_dt:.4f}s  duration={duration:.2f}s")
    return {"joint_pos": joint_pos, "frame_dt": frame_dt, "T": T, "duration": duration}


def load_motion_interpolated(npz_path: str) -> dict:
    data = np.load(npz_path)
    frame_dt = float(data["frame_duration"])
    raw_pyb = np.asarray(data["joint_pos"], dtype=np.float32)
    raw_isaac = raw_pyb[:, PYB_TO_ISAAC]
    
    clip_mean = raw_isaac.mean(axis=0)
    delta = (raw_isaac - clip_mean) * SIGN_FLIP[None, :]
    joint_pos = delta + ISAAC_DEFAULT[None, :]
    
    T = raw_pyb.shape[0]
    duration = T * frame_dt
    
    print(joint_pos, joint_pos[0])

    # 1. Append the first frame to the very end to create a closed loop
    joint_pos_looped = np.vstack((joint_pos, joint_pos[0]))
    
    # 2. Create the time array for the keyframes (length T + 1)
    t_frames = np.linspace(0, duration, T + 1)
    
    # 3. Fit the periodic cubic spline across all joints (axis=0)
    cs = CubicSpline(t_frames, joint_pos_looped, axis=0, bc_type='periodic')
    
    print(f"[Motion] {T} frames  dt={frame_dt:.4f}s  duration={duration:.2f}s")
    
    # Store the spline object directly in the clip dictionary
    return {
        "joint_pos": joint_pos, 
        "frame_dt": frame_dt, 
        "T": T, 
        "duration": duration,
        "cs": cs 
    }


# def get_joint_target(clip: dict, t: float) -> np.ndarray:
#     jp = clip["joint_pos"]
#     dt = clip["frame_dt"]
#     T = clip["T"]
#     u = (t / dt) % T
#     i0 = int(np.floor(u))
#     i1 = (i0 + 1) % T
#     a = float(u - i0)
#     return (1.0 - a) * jp[i0] + a * jp[i1]


# def get_joint_target_batch(clip: dict, t_arr: np.ndarray) -> np.ndarray:
#     """Vectorized version for N envs.  t_arr: (N,)  Returns: (N, 12)"""
#     jp = clip["joint_pos"]
#     dt = clip["frame_dt"]
#     T = clip["T"]
#     u = (t_arr / dt) % T  # (N,)
#     i0 = np.floor(u).astype(int)  # (N,)
#     i1 = (i0 + 1) % T
#     a = (u - i0)[:, None]  # (N,1)
#     return (1.0 - a) * jp[i0] + a * jp[i1]  # (N,12)


# def get_ref_joint_velocities_batch(clip: dict, t_arr: np.ndarray) -> np.ndarray:
#     jv = np.gradient(clip["joint_pos"], axis=0)
#     dt = clip["frame_dt"]
#     T = clip["T"]
#     u = (t_arr / dt) % T
#     i0 = np.floor(u).astype(int)
#     i1 = (i0 + 1) % T
#     a = (u - i0)[:, None]  # (N,1)
#     return (1.0 - a) * jv[i0] + a * jv[i1]


def get_joint_target_batch(clip: dict, t_arr: np.ndarray) -> np.ndarray:
    """Evaluate the cubic spline for smooth joint positions."""
    # Wrap the time array to stay within the clip duration
    t_wrapped = t_arr % clip["duration"]
    
    # Evaluate the spline at the queried times
    return clip["cs"](t_wrapped).astype(np.float32)

def get_ref_joint_velocities_batch(clip: dict, t_arr: np.ndarray) -> np.ndarray:
    """Evaluate the first derivative of the cubic spline for accurate joint velocities."""
    t_wrapped = t_arr % clip["duration"]
    
    # The '1' argument tells CubicSpline to return the 1st derivative
    return clip["cs"](t_wrapped, 1).astype(np.float32)


class MLP(nn.Module):
    """
    Simple MLP policy: obs → action.
    Used for both BC (deterministic) and RL (stochastic via log_std).
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden=(512, 256, 128), stochastic: bool = False):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(in_dim, action_dim)
        self.stochastic = stochastic
        if stochastic:
            self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mean = self.mean(h)
        if self.stochastic:
            std = self.log_std.exp().expand_as(mean)
            return torch.distributions.Normal(mean, std)
        return mean

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        if self.stochastic:
            dist = self.forward(obs)
            return dist.mean if deterministic else dist.rsample()
        return self.forward(obs)


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden=(256, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ELU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def make_env(num_envs: int, stiffness: float, damping: float):
    env_cfg = parse_env_cfg(
        task_name=args_cli.task, device="cuda:0", num_envs=num_envs)
    dc_cfg = ActuatorNetLSTMCfg(
            joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
            network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
            saturation_effort=120.0,
            effort_limit=80.0,
            velocity_limit=7.5,
        )
    env_cfg.scene.robot = env_cfg.scene.robot.replace(actuators={"legs": dc_cfg})
    env = gym.make(args_cli.task, cfg=env_cfg)
    return env


def init_joints_to_motion(env, clip: dict, phase_t_arr: np.ndarray, device: str):
    """Initialize all envs' joints to their respective motion phases."""
    robot: AssetBase = env.unwrapped.scene["robot"]
    num_envs = env.unwrapped.num_envs

    # Reset the joint states
    q_np = get_joint_target_batch(clip, phase_t_arr)  # (N,12)
    q_t = torch.tensor(q_np, dtype=torch.float32, device=device)
    v_t = torch.zeros_like(q_t)
    am = env.unwrapped.action_manager
    jids = am._terms["joint_pos"]._joint_ids
    robot.write_joint_state_to_sim(q_t, v_t, joint_ids=jids)

    # TODO: Reset the root states as well
    # s_t = robot.data.default_root_state.clone()  # Shape is (len(env_ids), 13). Refer to articulation.py base class 
    # robot.write_root_state_to_sim(root_state=s_t, env_ids=None) # all environments are used

    env.unwrapped.scene.write_data_to_sim()


def reference_action(clip: dict, t_arr: np.ndarray, default_pos: np.ndarray, device: str) -> torch.Tensor:
    """Compute reference actions for all envs."""
    q_np = get_joint_target_batch(clip, t_arr)  # (N,12)
    a_np = (q_np - default_pos[None, :]) / ACTION_SCALE
    return torch.tensor(a_np, dtype=torch.float32, device=device)


def collect_bc_data(env, clip: dict, device: str, collect_s: float = 60.0) -> tuple:
    """
    Roll out the reference motion for `collect_s` seconds per env.
    Returns (obs_buf, act_buf) as tensors on CPU.
    """
    robot = env.unwrapped.scene["robot"]
    num_envs = env.unwrapped.num_envs
    sim_dt = getattr(env.unwrapped, "step_dt", 0.02)
    default_pos = robot.data.default_joint_pos[0].cpu().numpy()
    n_steps = int(collect_s / sim_dt)

    obs_list = []
    act_list = []

    # Random starting phases for diversity
    phase_t = np.random.uniform(0, clip["duration"], size=num_envs)
    obs, _ = env.reset()
    obs = obs['policy']
    init_joints_to_motion(env, clip, phase_t, device)

    print(f"[BC collect] {n_steps} steps × {num_envs} envs = "
          f"{n_steps * num_envs:,} samples")

    for step in range(n_steps):
        acts = reference_action(clip, phase_t, default_pos, device)  # (N,12)
        obs_np = obs.cpu() if isinstance(obs, torch.Tensor) else torch.from_numpy(obs)

        obs_list.append(obs_np[:, :OBS_DIM])
        act_list.append(acts.cpu())

        obs, _, terminated, truncated, _ = env.step(acts)
        obs = obs['policy']
        phase_t += sim_dt

        # On reset: re-initialize joints to new random phases
        done = terminated | truncated
        if done.any():
            new_phases = np.random.uniform(0, clip["duration"], size=num_envs)
            phase_t[done.cpu().numpy()] = new_phases[done.cpu().numpy()]
            init_joints_to_motion(env, clip, phase_t, device)

        if step % 500 == 0:
            print(f"  step {step}/{n_steps}  "
                  f"collected {len(obs_list) * num_envs:,} samples")

    obs_buf = torch.cat(obs_list, dim=0)  # (N_total, 48)
    act_buf = torch.cat(act_list, dim=0)  # (N_total, 12)
    print(f"[BC collect] Done. Dataset: {obs_buf.shape}")
    return obs_buf, act_buf


def train_bc(obs_buf: torch.Tensor, act_buf: torch.Tensor, policy: MLP, device: str, epochs: int = 200, batch: int = 1024, lr: float = 3e-4, save_path: str = "policy_bc.pt"):
    """Train policy via supervised regression on (obs, action) pairs."""
    policy = policy.to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    N = obs_buf.shape[0]

    obs_d = obs_buf.to(device)
    act_d = act_buf.to(device)

    best_loss = float("inf")
    losses = deque(maxlen=50)
    loss_history = []   

    print(f"\n[BC train] {epochs} epochs  batch={batch}  lr={lr}")
    for epoch in range(epochs):
        # Shuffle
        idx = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch):
            b = idx[start:start + batch]
            pred = policy(obs_d[b])
            loss = nn.functional.mse_loss(pred, act_d[b])
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1
        sched.step()

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({"epoch": epoch, "loss": best_loss,
                        "state_dict": policy.state_dict()}, save_path)

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d}/{epochs}  loss={avg_loss:.5f}  "
                  f"best={best_loss:.5f}  lr={sched.get_last_lr()[0]:.2e}")

    print(f"[BC train] Best loss: {best_loss:.5f}  saved → {save_path}")
    return policy, loss_history


class PPOBuffer:
    """Minimal rollout buffer for PPO."""

    def __init__(self, num_envs: int, horizon: int, obs_dim: int, action_dim: int, device: str):
        self.obs = torch.zeros(horizon, num_envs, obs_dim, device=device)
        self.acts = torch.zeros(horizon, num_envs, action_dim, device=device)
        self.logps = torch.zeros(horizon, num_envs, device=device)
        self.rewards = torch.zeros(horizon, num_envs, device=device)
        self.dones = torch.zeros(horizon, num_envs, device=device)
        self.values = torch.zeros(horizon, num_envs, device=device)
        self.ptr = 0
        self.horizon = horizon

    def store(self, obs, acts, logps, rewards, dones, values):
        t = self.ptr
        self.obs[t] = obs
        self.acts[t] = acts
        self.logps[t] = logps
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        self.ptr = (self.ptr + 1) % self.horizon

    def compute_returns(self, last_value: torch.Tensor, gamma: float = 0.99, lam: float = 0.95):
        """GAE-Lambda returns."""
        H, N = self.horizon, self.obs.shape[1]
        adv = torch.zeros(H, N, device=self.obs.device)
        gae = torch.zeros(N, device=self.obs.device)
        for t in reversed(range(H)):
            next_val = last_value if t == H - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            adv[t] = gae
        returns = adv + self.values
        return adv.reshape(-1), returns.reshape(-1)

    def get_flat(self):
        return (self.obs.reshape(-1, self.obs.shape[-1]),
                self.acts.reshape(-1, self.acts.shape[-1]),
                self.logps.reshape(-1),
                self.values.reshape(-1))

# TODO: Define custom rewards that combine multiple different rewards instead of only MSE shadow to actual state
def motion_tracking_reward(joint_pos: torch.Tensor, 
                           q_ref: torch.Tensor, 
                           sigma: float = 0.25, 
                           vel_reward: torch.Tensor = None, 
                           vel_weight: float = 0.5, 
                           joint_vel_reward: torch.Tensor = None,
                           joint_vel_weight:float = 0.05) -> torch.Tensor:
    """
    Reward = exp(-||q - q_ref||^2 / sigma^2)
           + vel_weight * vel_reward (optional forward velocity term)

    joint_pos: (N, 12)  current joint positions
    q_ref:     (N, 12)  reference joint positions from motion clip
    """
    diff = joint_pos - q_ref  # (N, 12)
    sq_dist = (diff ** 2).sum(dim=-1)  # (N,)
    track_rew = torch.exp(-sq_dist / (sigma ** 2))  # (N,)


    if vel_reward is not None:
        track_rew += vel_weight * vel_reward.clamp(0) 

    if joint_vel_reward is not None:
        track_rew += joint_vel_weight * joint_vel_reward.clamp(0)

    return track_rew


def train_ppo(env, clip: dict, policy: MLP, value_net: ValueNet,
              device: str, n_iters: int = 2000, horizon: int = 24,
              lr: float = 1e-4, sigma: float = 0.25, vel_weight: float = 0.5,
              save_path: str = "policy_rl.pt",
              clip_eps: float = 0.2, entropy_coef: float = 0.01,
              vf_coef: float = 0.5, grad_clip: float = 1.0,
              n_epochs_per_iter: int = 4, minibatch: int = 512):
    robot = env.unwrapped.scene["robot"]
    num_envs = env.unwrapped.num_envs
    sim_dt = getattr(env.unwrapped, "step_dt", 0.02)
    default_pos = robot.data.default_joint_pos[0].cpu().numpy()
    default_t = torch.tensor(default_pos[:12], dtype=torch.float32, device=device)

    all_params = list(policy.parameters()) + list(value_net.parameters())
    optim = torch.optim.Adam(all_params, lr=lr)
    buf = PPOBuffer(num_envs, horizon, OBS_DIM, ACTION_DIM, device)

    phase_t = np.random.uniform(0, clip["duration"], size=num_envs)
    obs, _ = env.reset()
    obs = obs["policy"]
    init_joints_to_motion(env, clip, phase_t, device)

    best_mean_return = -float("inf")
    ep_returns = deque(maxlen=100)
    ep_return = torch.zeros(num_envs, device=device)

    print(f"\n[PPO] {n_iters} iters horizon={horizon}  lr={lr}")

    for iteration in range(n_iters):
        policy.eval()
        value_net.eval()

        with torch.no_grad():
            for _ in range(horizon):

                # TODO: Find another way of converting the types because inefficient
                if isinstance(obs, dict):
                    obs = obs['policy']
                    
                obs_t = obs[:, :OBS_DIM].to(device)
                dist = policy(obs_t)
                acts = dist.rsample()
                logps = dist.log_prob(acts).sum(-1)
                values = value_net(obs_t)

                next_obs, _, terminated, truncated, _ = env.step(acts)
                done = (terminated | truncated).float() #TODO: add more terminal conditions
                phase_t += sim_dt

                # Motion tracking reward
                q_cur = robot.data.joint_pos[:, :12].clone()  # (N,12)
                q_ref_np = get_joint_target_batch(clip, phase_t)
                q_ref = torch.tensor(q_ref_np, dtype=torch.float32, device=device)

                dq_cur = robot.data.joint_vel[:, :12].clone()  # (N,12)
                dq_ref_np = get_ref_joint_velocities_batch(clip, phase_t)
                dq_ref = torch.tensor(dq_ref_np, dtype=torch.float32, device=device)
               
                # Optional forward velocity reward from observations
                # obs[:, 0] = base_lin_vel_x (world frame, approx)
                vel_rew = torch.exp(-2 * (obs[:, 0] **  2)) #reward velocity component
                joint_vel_rew = torch.exp(-0.1 * ((dq_cur - dq_ref) ** 2).sum(dim=-1))

                reward = motion_tracking_reward(
                    q_cur, q_ref, sigma=sigma,
                    vel_reward=vel_rew, 
                    vel_weight=0.0,
                    joint_vel_reward=joint_vel_rew,
                    joint_vel_weight=0.0)
                

                # Directly the MDP into the IsaacLab Buffer, ignoring the Manager-based configurations
                buf.store(obs_t, acts, logps, reward, done, values)
                ep_return += reward

                # Handle resets
                reset_mask = done.bool().cpu().numpy()
                if reset_mask.any():
                    new_phases = np.random.uniform(0, clip["duration"], size=num_envs)
                    phase_t[reset_mask] = new_phases[reset_mask]
                    init_joints_to_motion(env, clip, phase_t, device)
                    for r in ep_return[done.bool()]:
                        ep_returns.append(r.item())
                    ep_return[done.bool()] = 0.0

                obs = next_obs

            # TODO: Find a more suitable way of converting obs instead of converting it from dict
            # Bootstrap last value
            if isinstance(obs, dict):
                obs = obs['policy']

            last_val = value_net(obs[:, :OBS_DIM].to(device))
            adv, ret = buf.compute_returns(last_val)

        policy.train()
        value_net.train()

        obs_flat, act_flat, old_logp_flat, _ = buf.get_flat()
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        total_pg_loss = total_vf_loss = total_ent = 0.0
        n_updates = 0

        for _ in range(n_epochs_per_iter):
            idx = torch.randperm(obs_flat.shape[0], device=device)
            for start in range(0, idx.shape[0], minibatch):
                b = idx[start:start + minibatch]
                dist = policy(obs_flat[b])
                new_lp = dist.log_prob(act_flat[b]).sum(-1)
                ratio = (new_lp - old_logp_flat[b]).exp()

                pg_loss = -torch.min(
                    ratio * adv_norm[b],
                    ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_norm[b]
                ).mean()
                vf_loss = nn.functional.mse_loss(value_net(obs_flat[b]), ret[b])
                ent = dist.entropy().sum(-1).mean()

                loss = pg_loss + vf_coef * vf_loss - entropy_coef * ent
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, grad_clip)
                optim.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent += ent.item()
                n_updates += 1

        mean_return = np.mean(ep_returns) if ep_returns else 0.0

        if mean_return > best_mean_return and ep_returns:
            best_mean_return = mean_return
            torch.save({"iter": iteration, "return": best_mean_return,
                        "policy": policy.state_dict(),
                        "value": value_net.state_dict()}, save_path)

        if iteration % 50 == 0 or iteration == n_iters - 1:
            print(f"  iter {iteration:5d}/{n_iters}  "
                  f"return={mean_return:.3f}  best={best_mean_return:.3f}  "
                  f"pg={total_pg_loss / n_updates:.4f}  "
                  f"vf={total_vf_loss / n_updates:.4f}  "
                  f"ent={total_ent / n_updates:.3f}")

    print(f"[PPO] Best mean return: {best_mean_return:.3f}  saved → {save_path}")


def evaluate(env, clip: dict, policy: MLP, device: str, n_episodes: int = 5):
    robot = env.unwrapped.scene["robot"]
    sim_dt = getattr(env.unwrapped, "step_dt", 0.02)
    default_pos = robot.data.default_joint_pos[0].cpu().numpy()
    num_envs = env.unwrapped.num_envs

    phase_t = np.zeros(num_envs)
    obs, _ = env.reset()
    obs = obs['policy']
    init_joints_to_motion(env, clip, phase_t, device)

    policy.eval()
    ep_returns = []
    ep_return = torch.zeros(num_envs, device=device)
    step = 0

    print(f"\n[Eval] Running {n_episodes} episodes...")

    while len(ep_returns) < n_episodes * num_envs:
        with torch.no_grad():
            if isinstance(obs, dict):
                obs = obs['policy']
            obs_t = obs[:, :OBS_DIM].to(device)
            acts = policy.act(obs_t, deterministic=True)

        obs, _, terminated, truncated, _ = env.step(acts)
        phase_t += sim_dt

        q_cur = robot.data.joint_pos[:, :12]
        q_ref_np = get_joint_target_batch(clip, phase_t)
        q_ref = torch.tensor(q_ref_np, dtype=torch.float32, device=device)
        reward = motion_tracking_reward(q_cur, q_ref)

        ep_return += reward
        done = (terminated | truncated).bool()
        if done.any():
            for r in ep_return[done]:
                ep_returns.append(r.item() / (step + 1))  # avg per-step return
            ep_return[done] = 0.0
            phase_t[done.cpu().numpy()] = 0.0
            init_joints_to_motion(env, clip, phase_t, device)

        step += 1

    mean_r = np.mean(ep_returns)
    std_r = np.std(ep_returns)
    print(f"[Eval] Mean per-step return: {mean_r:.4f} ± {std_r:.4f}")
    return mean_r

def plot_joint_motion(clip: dict, joint_idx: int = 0, num_cycles: int = 3, save_dir: str = "./checkpoints"):
    """
    Simulates high-resolution queries to the motion clip over multiple cycles
    to visualize the wrapping behavior and mathematical derivatives.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a high-resolution time array to see between the frames
    eval_dt = 0.005 # 5ms physics steps
    total_time = clip["duration"] * num_cycles
    t_eval = np.arange(0, total_time, eval_dt)
    
    # Query the spline for position and velocity using the wrapped time
    t_wrapped = (t_eval % clip["duration"])
    pos_eval = np.array(clip["joint_pos"])[t_wrapped.astype(int)]          # Position
    # vel_eval = clip["cs"](t_wrapped, 1)       # Velocity (1st derivative)
    vel_eval = get_ref_joint_velocities_batch(clip, t_eval)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. Plot Position
    ax1.plot(t_eval, pos_eval[:, joint_idx], label=f"Spline Position (Joint {joint_idx})", color='blue')
    
    # Overlay the original mocap keyframes for reference
    # We plot them repeatedly for each cycle
    t_frames = np.linspace(0, clip["duration"], clip["T"] + 1)
    for i in range(num_cycles):
        print(t_frames)
        ax1.plot(t_frames + (i * clip["duration"]), clip["joint_pos"][t_frames.astype(int)][:, joint_idx], 
                 'ro', markersize=4, label="Keyframes" if i == 0 else "")
                 
    # 2. Plot Velocity
    ax2.plot(t_eval, vel_eval[:, joint_idx], label=f"Spline Velocity (Joint {joint_idx})", color='green')
    
    # Draw vertical lines at the wrap boundaries
    for i in range(num_cycles + 1):
        for ax in [ax1, ax2]:
            ax.axvline(i * clip["duration"], color='gray', linestyle='--', alpha=0.5)
            
    ax1.set_ylabel("Position (rad)")
    ax1.set_title(f"Target Position over {num_cycles} Cycles (Checking Wrap Behavior)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel("Velocity (rad/s)")
    ax2.set_xlabel("Time (Seconds)")
    ax2.set_title("Target Velocity (Checking for Jitter/Spikes)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"joint_{joint_idx}_motion.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"[Diagnostic] Saved motion graph to {plot_path}")


def plot_joint_motion_cubic(clip: dict, joint_idx: int = 0, num_cycles: int = 3, save_dir: str = "./checkpoints"):
    """
    Simulates high-resolution queries to the motion clip over multiple cycles
    to visualize the wrapping behavior and mathematical derivatives.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a high-resolution time array to see between the frames
    eval_dt = 0.005 # 5ms physics steps
    total_time = clip["duration"] * num_cycles
    t_eval = np.arange(0, total_time, eval_dt)
    
    # Query the spline for position and velocity using the wrapped time
    t_wrapped = (t_eval % clip["duration"])
    pos_eval = clip["cs"](t_wrapped)          # Position
    # vel_eval = clip["cs"](t_wrapped, 1)       # Velocity (1st derivative)
    vel_eval = get_ref_joint_velocities_batch(clip, t_eval)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. Plot Position
    ax1.plot(t_eval, pos_eval[:, joint_idx], label=f"Spline Position (Joint {joint_idx})", color='blue')
    
    # Overlay the original mocap keyframes for reference
    # We plot them repeatedly for each cycle
    t_frames = np.linspace(0, clip["duration"], clip["T"] + 1)
    for i in range(num_cycles):
        print(t_frames)
        ax1.plot(t_frames + (i * clip["duration"]), clip["cs"](t_frames.astype(int))[:, joint_idx], 
                 'ro', markersize=4, label="Keyframes" if i == 0 else "")
                 
    # 2. Plot Velocity
    ax2.plot(t_eval, vel_eval[:, joint_idx], label=f"Spline Velocity (Joint {joint_idx})", color='green')
    
    # Draw vertical lines at the wrap boundaries
    for i in range(num_cycles + 1):
        for ax in [ax1, ax2]:
            ax.axvline(i * clip["duration"], color='gray', linestyle='--', alpha=0.5)
            
    ax1.set_ylabel("Position (rad)")
    ax1.set_title(f"Target Position over {num_cycles} Cycles (Checking Wrap Behavior)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel("Velocity (rad/s)")
    ax2.set_xlabel("Time (Seconds)")
    ax2.set_title("Target Velocity (Checking for Jitter/Spikes)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"joint_{joint_idx}_motion.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"[Diagnostic] Saved motion graph to {plot_path}")


def main():
    device = "cuda:0"
    # clip = load_motion(args_cli.motion)
    clip = load_motion_interpolated(args_cli.motion)

    plot_joint_motion_cubic(clip, joint_idx=0, num_cycles=3)

    # Single env for eval, multiple for training
    num_envs = 1 if args_cli.mode == "eval" else args_cli.num_envs
    env = make_env(num_envs, args_cli.stiffness, args_cli.damping)

    if args_cli.mode == "bc":
        policy = MLP(OBS_DIM, ACTION_DIM, stochastic=False)
        save_path = os.path.join(args_cli.save_dir, "policy_bc.pt")

        obs_buf, act_buf = collect_bc_data(
            env, clip, device, collect_s=args_cli.bc_collect_s)

        policy, loss_history = train_bc(
            obs_buf, act_buf, policy, device,
            epochs=args_cli.bc_epochs,
            batch=args_cli.bc_batch,
            lr=args_cli.bc_lr,
            save_path=save_path,
        )

        print(f"\n[BC] Training complete.  Evaluating trained policy...")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(loss_history) + 1), loss_history, label='MSE Loss', color='b', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Mean Squared Error', fontsize=12)
        plt.title('Behavioral Cloning Training Loss', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        plot_path = os.path.join(args_cli.save_dir, "bc_loss_curve.png")
        plt.savefig(plot_path, dpi=300)
        plt.close() # Close the figure to free up memory
        print(f"[BC] Saved loss curve plot to {plot_path}")

        print(f"\n[BC] Training complete.  Evaluating trained policy...")
        evaluate(env, clip, policy.to(device), device)

    elif args_cli.mode == "rl":
        policy = MLP(OBS_DIM, ACTION_DIM, stochastic=True)
        value_net = ValueNet(OBS_DIM)
        save_path = os.path.join(args_cli.save_dir, "policy_rl.pt")

        # Optionally warm-start from BC checkpoint
        bc_ckpt = args_cli.bc_ckpt or args_cli.ckpt
        if bc_ckpt and os.path.exists(bc_ckpt):
            ck = torch.load(bc_ckpt, map_location=device)
            sd = ck.get("state_dict") or ck.get("policy")
            policy.load_state_dict(sd, strict=False)
            print(f"[RL] Loaded BC init from {bc_ckpt}")
        elif bc_ckpt:
            print(f"[RL] Warning: checkpoint {bc_ckpt} not found, training from scratch.")

        train_ppo(
            env, clip, policy.to(device), value_net.to(device), device,
            n_iters=args_cli.rl_iters,
            horizon=args_cli.rl_steps,
            lr=args_cli.rl_lr,
            sigma=args_cli.rl_sigma,
            vel_weight=args_cli.rl_vel_weight,
            save_path=save_path,
        )

    elif args_cli.mode == "eval":
        if not args_cli.ckpt:
            raise ValueError("--ckpt required for eval mode")

        ck = torch.load(args_cli.ckpt, map_location=device)
        # Auto-detect BC vs RL checkpoint
        is_rl = "policy" in ck
        policy = MLP(OBS_DIM, ACTION_DIM, stochastic=is_rl)
        sd = ck.get("policy") or ck.get("state_dict")
        policy.load_state_dict(sd)
        policy = policy.to(device)
        print(f"[Eval] Loaded {'RL' if is_rl else 'BC'} policy from {args_cli.ckpt}")

        evaluate(env, clip, policy, device, n_episodes=10)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
