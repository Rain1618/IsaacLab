import argparse
from isaaclab.app import AppLauncher  

parser = argparse.ArgumentParser(description="ANYmal-D motion playback in Isaac Lab.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Flat-Anymal-D-v0",
    help="Isaac Lab task name.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of parallel envs.",
)
parser.add_argument(
    "--motion",
    type=str,
    default="/home/rainagulce/Documents/IsaacLab/scripts/environments/data/pace.npz",  
    help="Path to the ANYmal motion .npz file.",
)
args_cli = parser.parse_args()

# launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# After app is running, we can import Isaac Lab / Isaac Sim modules
import gymnasium as gym
import torch
import numpy as np

import isaaclab_tasks  # noqa: F401  # registers all envs
from isaaclab_tasks.utils import parse_env_cfg  

import numpy as np
import torch

class MotionToIsaacActionConverter:
    """
    Converts PyBullet-style absolute joint angles from your reference motion
    into Isaac Lab ANYmal-D actions (normalized, ordered, batched).

    Assumes:
    - 12 actuated joints for ANYmal-D.
    - env.action_space is (12,) and uses normalized actions.
    - player.joint_pos is (T, 12) in PyBullet joint order:
        [LF_HAA, LF_HFE, LF_KFE,
         RF_HAA, RF_HFE, RF_KFE,
         LH_HAA, LH_HFE, LH_KFE,
         RH_HAA, RH_HFE, RH_KFE]
    """

    def __init__(self, env, player, device=None):
        self.env = env
        self.player = player
        self.device = device if device is not None else env.unwrapped.device

        self.num_envs = env.unwrapped.num_envs
        self.n_act = env.action_space.shape[-1]

        # Isaac DOF order:
        # [LF_HIP, LH_HIP, RF_HIP, RH_HIP,
        #  LF_THIGH, LH_THIGH, RF_THIGH, RH_THIGH,
        #  LF_SHANK, LH_SHANK, RF_SHANK, RH_SHANK]
        #
        # PyBullet order:
        # [LF_HAA, LF_HFE, LF_KFE,
        #  RF_HAA, RF_HFE, RF_KFE,
        #  LH_HAA, LH_HFE, LH_KFE,
        #  RH_HAA, RH_HFE, RH_KFE]
        #
        # So Isaac index -> PyBullet index:

        self.pyb_from_isaac = np.array(
            [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11],
            dtype=int,
        )

        # Hard-coded ANYmal-D default pose in Isaac DOF order
        # (hip = 0, thigh ≈ ±0.4, knee ≈ ∓0.8)
        self.joint_offsets = np.array(
            [0.0, 0.0, 0.0, 0.0,
             0.4, -0.4, 0.4, -0.4,
            -0.8, 0.8, -0.8, 0.8],
            dtype=np.float32,
        )

        # Precompute per-joint scale from the entire motion clip so that
        # the largest deviation from default maps to about action = ±1.
        self.joint_scale = self._compute_joint_scale_from_clip()
        print("[INFO] joint_scale from clip:", self.joint_scale)

        # Unwrap the env manually and get the action configurations
        # self.action_cfg = self.env.unwrapped.action_manager.actions["joint_pos"]
        # self.action_scale = self.action_cfg.scale

        # self.robot = self.env.unwrapped.scene["robot"]


        # Patch internal action buffer if needed (fixes the [0,12] bug)
        am = env.unwrapped.action_manager
        if am._action.shape[0] == 0:
            am._action = torch.zeros(
                (self.num_envs, self.n_act), device=self.device
            )

    def _compute_joint_scale_from_clip(self):
        """Compute per-joint normalization factors from the whole clip."""
        all_q_pyb = self.player.joint_pos              # (T, 12) PyBullet order
        all_q_isaac = all_q_pyb[:, self.pyb_from_isaac]  # (T, 12)

        delta = all_q_isaac - self.joint_offsets[None, :]  # (T, 12)
        scale = np.max(np.abs(delta), axis=0)              # (12,)
        # Avoid zero scales
        scale = np.maximum(scale, 0.1)
        return scale.astype(np.float32)

    def ref_to_action(self, q_ref_pyb, action_scale=1.0):
        """
        Convert one frame of PyBullet joint angles (shape (12,))
        into a batched Isaac action tensor (num_envs, 12) on the right device.

        TRY SIMPLIFYING THIS AND UNDERSTAND HOW ISAAC LAB PROCESSES ACTIONS
        """
        # 1) PyBullet → Isaac joint order
        q_ref_isaac = q_ref_pyb[self.pyb_from_isaac]              # (12,)
        # q_ref_isaac[:] = 0

        # 2) Absolute → offset from default pose
        delta = q_ref_isaac                                         # (12,)

        # 3) Normalize per-joint using motion-derived scales
        # action_np = (delta * action_scale) % (2 * np.pi)                   # (12,)

        # Wrap the joint movements
        # action_np = self.wrap_to_pi(action_np)

        # 4) Clamp to action space limits
        # low = self.env.action_space.low
        # high = self.env.action_space.high

        # action_np = np.clip(action_np, low, high)                 # (12,)

        # 5) Broadcast to all envs: (num_envs, 12)
        # actions_np = np.tile(action_np, (self.num_envs, 1))
        actions_np = np.tile(delta, (self.num_envs, 1))


        print(actions_np)

        # 6) Torch tensor on correct device
        actions = torch.from_numpy(actions_np).to(self.device)

        return actions
    
    @staticmethod
    def wrap_to_pi(angles):
        """
        Wraps angles to the range [-pi, pi).
        """
        return (angles + np.pi) % (2 * np.pi) - np.pi


class AnymalMotionPlayer:
    def __init__(
        self,
        npz_path: str,
        joint_key: str = "joint_pos",
        dt_key: str = "frame_duration",
    ):
        data = np.load(npz_path)
        print("[MotionPlayer] keys in npz:", data.files)

        self.frame_dt = float(data[dt_key])
        self.joint_pos = np.asarray(data[joint_key], dtype=np.float32)  # (T, 12)
        self.num_frames, self.num_joints = self.joint_pos.shape

        print(
            f"[MotionPlayer] Loaded {self.num_frames} frames, "
            f"{self.num_joints} joints, dt={self.frame_dt:.4f}s"
        )

    def get_joint_targets(self, t: float) -> np.ndarray:
        """Return interpolated joint positions at time t (looping)."""
        if self.num_frames == 0:
            return np.zeros(self.num_joints, dtype=np.float32)

        u = (t / self.frame_dt) % self.num_frames
        i0 = int(np.floor(u))
        i1 = (i0 + 1) % self.num_frames
        alpha = float(u - i0)

        q0 = self.joint_pos[i0]
        q1 = self.joint_pos[i1]
        return (1.0 - alpha) * q0 + alpha * q1

def main():
    # 1. Build env and reset
    env_cfg = parse_env_cfg(
        task_name=args_cli.task,
        device="cuda:0",
        num_envs=args_cli.num_envs,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO] Obs space:   {env.observation_space}")
    print(f"[INFO] Action space:{env.action_space}")
    num_envs = env.unwrapped.num_envs
    n_act = env.action_space.shape[-1]
    print(f"[INFO] num_envs from env: {num_envs}, n_act: {n_act}")
    robot = env.unwrapped.scene["robot"]

    obs, info = env.reset()
    player = AnymalMotionPlayer(args_cli.motion)
    converter = MotionToIsaacActionConverter(env, player)

    # Configure the env

    # env.action_space.low = -np.pi
    # env.action_space.high = np.pi

    sim_dt = getattr(env.unwrapped, "step_dt", 0.02)
    sim_time = 0.0
    time_scale = 1.0   # <1.0 = slower, smoother
    action_scale = 0.5  # refer to ActionCfg in velocity_env_cfg.py

    joint_limits = robot.data.soft_joint_pos_limits[0]

    for i, name in enumerate(robot.data.joint_names):
        lower = joint_limits[i, 0].item()
        upper = joint_limits[i, 1].item()
        print(f"[DEBUG] Joint: {name} | Lower: {lower} | Upper: {upper}")

    while simulation_app.is_running():
        with torch.inference_mode():
            # PyBullet-ordered joint angles (interpolated)
            q_ref_pyb = player.get_joint_targets(sim_time)  

            # Convert to batched Isaac actions
            actions = converter.ref_to_action(q_ref_pyb, action_scale=action_scale)

            obs, reward, terminated, truncated, info = env.step(actions)
            
            if terminated.any() or truncated.any():
                sim_time = 0.0
                print("[DEBUG] Episode reset at sim_time", sim_time)
            
            if int(sim_time / sim_dt) % 50 == 0:  # every 50 steps or so
                dof_pos = robot.data.joint_pos[0].cpu().numpy()  # all DOFs
                # skip index 0 (base), print 12 leg joints
                print("[DEBUG] joint_pos[1:13]:", dof_pos[1:13])

            print("[DEBUG] joint_pos[1:13]:", dof_pos[1:13])

        sim_time += sim_dt * time_scale

    env.close()




if __name__ == "__main__":
    main()
    simulation_app.close()
