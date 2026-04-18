import argparse
import time
from pathlib import Path

import numpy as np
import torch

ISAAC_TO_PYB = np.array([0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11], dtype=np.int64)
SIGN_FLIP = np.ones(12, dtype=np.float32)
SIGN_FLIP[2] = -1.0
SIGN_FLIP[3] = -1.0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--ref-motion", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--physics_dt", type=float, default=1.0 / 120.0)
    p.add_argument("--ik_damping", type=float, default=0.05)
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--robot", type=str, default="anymal_d")
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--height_offset", type=float, default=0.0)
    p.add_argument("--ik_iterations", type=int, default=20)
    p.add_argument("--visualise", action="store_true", default=False)
    return p


parser = _build_arg_parser()

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = not args.visualise
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, Articulation, ArticulationCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from helpers import (  # noqa: E402
    ROBOT_META,
    QuadrupedRetargeter,
    compute_scale_factors,
    ensure_quaternion_continuity,
    get_robot_cfg,
    load_reference_motion,
    local_to_world_points,
    world_to_local_points,
)


@configclass
class RetargetSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot: ArticulationCfg = None


def smooth_signal(x: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1 or x.shape[0] < 3:
        return x.astype(np.float32)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / float(window)
    xpad = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    out = np.empty_like(x, dtype=np.float32)
    for idx in np.ndindex(x.shape[1:]):
        out[(slice(None),) + idx] = np.convolve(xpad[(slice(None),) + idx], kernel, mode="valid")
    return out.astype(np.float32)


def build_scaled_targets(ref, meta: dict, scale: dict, height_offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root_rot = ensure_quaternion_continuity(ref.root_rot)

    foot_local_ref = world_to_local_points(ref.root_pos, root_rot, ref.foot_pos)

    # already smoothed
    foot_local_scaled = foot_local_ref.copy()
    foot_local_scaled[..., 0] *= scale["x"]
    foot_local_scaled[..., 1] *= scale["y"]
    foot_local_scaled[..., 2] *= scale["z"]
    foot_local_scaled[..., 2] = np.clip(foot_local_scaled[..., 2], -meta["leg_length"] * 1.05, -0.02)

    scaled_root_pos = np.zeros_like(ref.root_pos, dtype=np.float32)
    scaled_root_pos[:, :2] = ref.root_pos[:, :2]
    scaled_root_pos[:, 2] = meta["base_height"] + float(height_offset)

    scaled_foot_pos_world = local_to_world_points(scaled_root_pos, root_rot, foot_local_scaled)

    return scaled_root_pos.astype(np.float32), root_rot.astype(np.float32), scaled_foot_pos_world.astype(np.float32)


def compute_kinematics_from_positions(joint_pos: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    joint_pos_s = smooth_signal(joint_pos, window=7)
    joint_vel = np.gradient(joint_pos_s, dt, axis=0).astype(np.float32)
    joint_vel = smooth_signal(joint_vel, window=7)
    joint_acc = np.gradient(joint_vel, dt, axis=0).astype(np.float32)
    joint_acc = smooth_signal(joint_acc, window=7)
    return joint_vel, joint_acc


def to_training_joint_format(joint_pos_isaac: np.ndarray, isaac_default: np.ndarray) -> np.ndarray:
    raw_isaac_export = (joint_pos_isaac - isaac_default[None, :]) / SIGN_FLIP[None, :]
    return raw_isaac_export[:, ISAAC_TO_PYB].astype(np.float32)


def run_targeting(args) -> None:
    device = args.device
    num_envs = args.num_envs
    if num_envs != 1:
        raise ValueError("Retarget export currently expects --num_envs 1.")

    print("\n[Retarget] Initializing...")
    ref = load_reference_motion(args.ref_motion, args.fps)
    meta = ROBOT_META[args.robot]
    scale = compute_scale_factors(ref, meta)
    scaled_root_pos, scaled_root_rot, scaled_foot_pos_world = build_scaled_targets(ref, meta, scale, args.height_offset)
    scaled_foot_pos_local = world_to_local_points(scaled_root_pos, scaled_root_rot, scaled_foot_pos_world)

    sim_cfg = sim_utils.SimulationCfg(
        dt=args.physics_dt,
        render_interval=1 if args.visualise else 4,
        device=device,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    if args.visualise:
        sim.set_camera_view(eye=[2.5, 2.0, 1.6], target=[0.0, 0.0, 0.45])

    robot_cfg = get_robot_cfg(args.robot)
    scene_cfg = RetargetSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene_cfg.robot = robot_cfg
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    print(f"\n[INFO] Setup Complete...")

    robot: Articulation = scene["robot"]
    scene.update(args.physics_dt)
    robot.update(args.physics_dt)

    if args.visualise:
        marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/foot_targets")
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        foot_markers = VisualizationMarkers(marker_cfg)
    else:
        foot_markers = None

    retargeter = QuadrupedRetargeter(
        robot=robot,
        scene=scene,
        sim=sim,
        meta=meta,
        device=device,
        ik_iterations=args.ik_iterations,
        ik_damping=args.ik_damping,
        physics_dt=args.physics_dt,
        num_envs=num_envs,
    )

    # get the total timesteps from inferring the horizontal length of the arr
    T = ref.root_pos.shape[0]
    isaac_joint_names = list(robot.joint_names)
    isaac_default = retargeter.default_qpos[0].detach().cpu().numpy().astype(np.float32)

    out_root_pos = scaled_root_pos.astype(np.float32)
    out_root_rot = scaled_root_rot.astype(np.float32)
    
    out_joint_pos = np.zeros((T, len(isaac_joint_names)), dtype=np.float32)
    out_foot_pos = np.zeros((T, 4, 3), dtype=np.float32)

    print(f"[Retarget] Processing {T} frames on {device}...")
    start = time.time()
    for t in range(T):
        jpos, _ = retargeter.solve_frame(
            root_pos=out_root_pos[t],
            root_rot=out_root_rot[t],
            foot_pos_local=scaled_foot_pos_local[t],
        )
        out_joint_pos[t] = jpos
        robot.update(args.physics_dt)
        for leg_i, foot_id in enumerate(retargeter.foot_body_ids):
            out_foot_pos[t, leg_i] = robot.data.body_pos_w[0, foot_id].detach().cpu().numpy()

        if foot_markers is not None:
            foot_markers.visualize(translations=torch.tensor(scaled_foot_pos_world[t], device=device, dtype=torch.float32))
            sim.render()

        if (t + 1) % 25 == 0 or t == T - 1:
            elapsed = max(time.time() - start, 1e-6)
            rate = (t + 1) / elapsed
            print(f"  frame {t + 1:5d}/{T} | {rate:.1f} frames/s")

    out_joint_vel, out_joint_acc = compute_kinematics_from_positions(out_joint_pos, ref.dt)
    joint_pos_for_training = to_training_joint_format(out_joint_pos, isaac_default)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        root_pos=out_root_pos,
        root_rot=out_root_rot,
        root_lin_vel=out_root_lin_vel,
        root_ang_vel=out_root_ang_vel,
        joint_pos_isaac=out_joint_pos,
        joint_vel=out_joint_vel,
        joint_acc=out_joint_acc,
        foot_pos=out_foot_pos,
        foot_pos_local=scaled_foot_pos_local.astype(np.float32),
        frame_duration=np.array(ref.dt, dtype=np.float32),
        joint_pos=joint_pos_for_training,
        fps=np.array(args.fps, dtype=np.float32),
        robot=np.array([args.robot]),
        joint_names=np.array(isaac_joint_names),
        foot_names=np.array(meta["foot_bodies"]),
        dt=np.array(ref.dt, dtype=np.float32),
        isaac_default_joint_pos=isaac_default,
        scale_x=np.array(scale["x"], dtype=np.float32),
        scale_y=np.array(scale["y"], dtype=np.float32),
        scale_z=np.array(scale["z"], dtype=np.float32),
    )
    print(f"[Retarget] Saved retargeted trajectory -> '{out_path}'")


if __name__ == "__main__":
    try:
        run_targeting(args)
    finally:
        simulation_app.close()
