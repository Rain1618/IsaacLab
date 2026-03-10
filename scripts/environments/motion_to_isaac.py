import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Anymal-D-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--motion", type=str, required=True)
parser.add_argument("--stiffness", type=float, default=200.0)
parser.add_argument("--damping", type=float, default=5.0)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--random_phase", action="store_true",
                    help="Start each episode at a random phase in the motion clip. Recommended for RL training data diversity.")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.actuators import DCMotorCfg

# ToDo: you can get it from this: env.unwrapped.scene["robot"].data.default_joint_pos[0].cpu().numpy()
ISAAC_DEFAULT = np.array([
    0.0, 0.0, 0.0, 0.0,  # HAA: LF LH RF RH
    0.4, -0.4, 0.4, -0.4,  # HFE: LF LH RF RH
    -0.8, 0.8, -0.8, 0.8,  # KFE: LF LH RF RH
], dtype=np.float32)

PYB_TO_ISAAC = np.array(
    [0, 6, 3, 9,  # HAA
     1, 7, 4, 10,  # HFE
     2, 8, 5, 11],  # KFE
    dtype=int,
)

ISAAC_JOINT_NAMES = [
    "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
    "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
    "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE",
]

SIGN_FLIP = np.ones(12, dtype=np.float32)
SIGN_FLIP[2] = -1.0  # RF_HAA
SIGN_FLIP[3] = -1.0  # RH_HAA


def load_motion(npz_path: str) -> dict:
    data = np.load(npz_path)
    frame_dt = float(data["frame_duration"])
    raw_pyb = np.asarray(data["joint_pos"], dtype=np.float32)
    raw_isaac = raw_pyb[:, PYB_TO_ISAAC]
    clip_mean = raw_isaac.mean(axis=0)
    delta = (raw_isaac - clip_mean) * SIGN_FLIP[None, :]
    joint_pos = delta + ISAAC_DEFAULT[None, :]
    T = raw_pyb.shape[0]
    print(f"[Motion] {T} frames  dt={frame_dt:.4f}s  duration={T * frame_dt:.2f}s")
    return {"joint_pos": joint_pos, "frame_dt": frame_dt, "T": T}


def get_joint_target(clip: dict, t: float) -> np.ndarray:
    jp = clip["joint_pos"]
    dt = clip["frame_dt"]
    T = clip["T"]
    u = (t / dt) % T
    i0 = int(np.floor(u))
    i1 = (i0 + 1) % T
    a = float(u - i0)
    return (1.0 - a) * jp[i0] + a * jp[i1]


def patch_actuator(env_cfg, stiffness: float, damping: float):
    # Following is from: source/isaaclab_assets/isaaclab_assets/robots/anymal.py
    dc_cfg = DCMotorCfg(
        joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
        saturation_effort=120.0,
        effort_limit=80.0,
        velocity_limit=7.5,
        stiffness={".*": stiffness},
        damping={".*": damping},
    )
    env_cfg.scene.robot = env_cfg.scene.robot.replace(actuators={"legs": dc_cfg})
    print(f"[INFO] Actuator: DCMotor  k={stiffness}  d={damping}")
    return env_cfg


def initialize_robot_to_motion(env, clip: dict, phase_t: float, device: str):
    """
    After env.reset(), write the motion frame's joint positions directly
    into the robot's state so the PD controller starts tracking from the
    correct position with zero transient.

    This replaces the random joint initialization from reset_joints_by_scale.
    """
    robot = env.unwrapped.scene["robot"]
    num_envs = env.unwrapped.num_envs

    q_target = get_joint_target(clip, phase_t)  # (12,)
    q_tensor = torch.tensor(q_target, dtype=torch.float32, device=device).unsqueeze(0).expand(num_envs, -1)  # (N,12)

    # Also set zero joint velocities so the robot isn't spinning on spawn
    v_tensor = torch.zeros_like(q_tensor)

    # Get the joint indices controlled by the action term
    am = env.unwrapped.action_manager
    term = am._terms["joint_pos"]
    jids = term._joint_ids

    # Write state directly bypasses the action buffer
    robot.write_joint_state_to_sim(q_tensor, v_tensor, joint_ids=jids)

    # Propagate to physics
    env.unwrapped.scene.write_data_to_sim()

    print(f"[Reset] Initialized joints to motion t={phase_t:.3f}s")


def main():
    clip = load_motion(args_cli.motion)
    env_cfg = parse_env_cfg(task_name=args_cli.task, device="cuda:0", num_envs=args_cli.num_envs, )
    env_cfg = patch_actuator(env_cfg, args_cli.stiffness, args_cli.damping)
    env = gym.make(args_cli.task, cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device

    am = env.unwrapped.action_manager
    term = am._terms["joint_pos"]
    default_pos = robot.data.default_joint_pos[0].cpu().numpy()
    action_scale = 0.5
    num_envs = env.unwrapped.num_envs
    sim_dt = getattr(env.unwrapped, "step_dt", 0.02)

    # Choose starting phase
    if args_cli.random_phase:
        phase_t = np.random.uniform(0, clip["T"] * clip["frame_dt"])
    else:
        phase_t = 0.0

    obs, _ = env.reset()

    # Initialize joints to the motion's starting frame immediately after reset
    initialize_robot_to_motion(env, clip, phase_t, device)

    sim_time = phase_t
    step = 0
    resets = 0

    print(f"\n[INFO] k={args_cli.stiffness}  speed={args_cli.speed}x  "
          f"sim_dt={sim_dt:.4f}s  start_phase={phase_t:.3f}s\n")

    while simulation_app.is_running():
        with torch.inference_mode():
            q_target = get_joint_target(clip, sim_time)
            # reverse the scaling and centering from the `process_actions()` to get the "correct" target joint positions
            action_np = (q_target - default_pos[:12]) / action_scale
            actions = torch.tensor(np.tile(action_np, (num_envs, 1)), device=device, dtype=torch.float32)

            obs, reward, terminated, truncated, info = env.step(actions)

            if step < 5 or step % 200 == 0:
                actual = robot.data.joint_pos[0].cpu().numpy()
                print(f"\n[step={step:4d}  t={sim_time:.3f}s  resets={resets}]")
                print(f"  {'joint':>8}  {'target':>8}  {'actual':>8}  {'error':>8}")
                for i in range(12):
                    err = q_target[i] - actual[i]
                    flag = "  <-" if abs(err) > 0.15 else ""
                    print(f"  {ISAAC_JOINT_NAMES[i]:>8}  {q_target[i]:+8.4f}  "
                          f"{actual[i]:+8.4f}  {err:+8.4f}{flag}")

            if terminated.any() or truncated.any():
                resets += 1
                obs, _ = env.reset()

                # Choose next episode's starting phase
                if args_cli.random_phase:
                    phase_t = np.random.uniform(0, clip["T"] * clip["frame_dt"])
                else:
                    phase_t = 0.0

                # Re-initialize joints to avoid post-reset transient
                initialize_robot_to_motion(env, clip, phase_t, device)

                sim_time = phase_t
                step = 0
                print(f"\n[RESET #{resets}  new_phase={phase_t:.3f}s]")
                continue

        sim_time += sim_dt * args_cli.speed
        step += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
