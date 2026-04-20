import argparse
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt  # noqa: E402

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
    p.add_argument("--plot_skeleton", action="store_true", default=False)
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
from isaaclab.utils import configclass  # noqa: E402f

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

class LiveSkeletonPlotter:
    """Live 3D skeleton plot in the robot's body frame (normalized coords).

    Root sits at the origin; feet are drawn at their body-frame positions.
    Axis limits are fixed from the full precomputed trajectory so the view
    doesn't jitter during playback.
    """

    FOOT_EDGES = [0, 1, 2, 3]  # root -> each of the four feet

    def __init__(self, foot_pos_local_all: np.ndarray, foot_names: list[str], title: str = "Retarget skeleton (body frame)"):
        plt.ion()
        self._fig = plt.figure(figsize=(7, 7))
        self._ax = self._fig.add_subplot(111, projection="3d")
        self._ax.view_init(elev=20.0, azim=-100.0)
        self._title = title
        self._foot_names = foot_names

        # Precompute fixed axis bounds from the whole trajectory.
        # Include the origin (root) so the frame is always visible.
        all_pts = np.concatenate(
            [np.zeros((1, 3), dtype=np.float32), foot_pos_local_all.reshape(-1, 3)],
            axis=0,
        )
        finite = np.isfinite(all_pts).all(axis=1)
        pts = all_pts[finite]
        mins, maxs = pts.min(axis=0), pts.max(axis=0)
        center = 0.5 * (mins + maxs)
        half = 0.5 * float(np.max(maxs - mins))
        half = max(half, 1e-3) * 1.1  # 10% padding
        self._bounds = {
            "x": (float(center[0] - half), float(center[0] + half)),
            "y": (float(center[1] - half), float(center[1] + half)),
            "z": (float(center[2] - half), float(center[2] + half)),
        }

    def update(self, root_world_t: np.ndarray, foot_local_t: np.ndarray, frame_idx: int, t_sec: float) -> None:
        """Redraw one frame. foot_local_t has shape (4, 3) in body frame."""
        ax = self._ax
        ax.clear()

        if root_world_t is None:
            root = np.zeros(3, dtype=np.float32)
        else: 
            root = root_world_t 

        # Scatter root and feet
        ax.scatter(root[0], root[1], root[2], s=90, label="root")
        ax.scatter(foot_local_t[:, 0], foot_local_t[:, 1], foot_local_t[:, 2], s=50, label="feet")

        # Edges from root to each foot
        for foot_i in self.FOOT_EDGES:
            fp = foot_local_t[foot_i]
            ax.plot([root[0], fp[0]], [root[1], fp[1]], [root[2], fp[2]], linewidth=2)

        # Labels
        for i, name in enumerate(self._foot_names):
            ax.text(foot_local_t[i, 0], foot_local_t[i, 1], foot_local_t[i, 2], name, fontsize=7)

        # Fixed formatting
        ax.set_xlabel("x (body)")
        ax.set_ylabel("y (body)")
        ax.set_zlabel("z (body)")
        ax.set_xlim(self._bounds["x"])
        ax.set_ylim(self._bounds["y"])
        ax.set_zlim(self._bounds["z"])
        xr = self._bounds["x"][1] - self._bounds["x"][0]
        yr = self._bounds["y"][1] - self._bounds["y"][0]
        zr = self._bounds["z"][1] - self._bounds["z"][0]
        try:
            ax.set_box_aspect((xr, yr, zr))
        except Exception:
            pass

        ax.set_title(f"{self._title} | frame={frame_idx} | t={t_sec:.2f}s")

        # Non-blocking redraw
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def close(self) -> None:
        plt.ioff()
        try:
            plt.close(self._fig)
        except Exception:
            pass


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

def build_scaled_targets(root_pos: np.ndarray,
                         root_rot: np.ndarray,
                         foot_pos: np.ndarray,
                         meta: dict,
                         scale: dict,
                         height_offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (scaled_root_pos, root_rot, scaled_foot_pos_world, foot_local_scaled).

    `foot_local_scaled` is expressed in the *original* reference root frame
    (from the un-modified root_pos / root_rot) so that feet motion stays
    relative to the dog's body even when the robot base is pinned.
    """
    root_rot = ensure_quaternion_continuity(root_rot)

    # Feet relative to the ORIGINAL reference root frame.
    foot_local_ref = world_to_local_points(root_pos, root_rot, foot_pos)

    foot_local_scaled = foot_local_ref.copy()
    foot_local_scaled[..., 0] *= scale["x"]
    foot_local_scaled[..., 1] *= scale["y"]
    foot_local_scaled[..., 2] *= scale["z"]
    # foot_local_scaled[..., 2] = np.clip(foot_local_scaled[..., 2], -meta["leg_length"] * 1.05, -0.02)

    # Kept for visualization / logging: world-frame feet under the (original)
    # reference root, lifted by the robot's base height + offset.
    scaled_root_pos = root_pos.astype(np.float32).copy()

    scaled_foot_pos_world = local_to_world_points(scaled_root_pos, root_rot, foot_local_scaled)

    return (
        scaled_root_pos.astype(np.float32),
        root_rot.astype(np.float32),
        scaled_foot_pos_world.astype(np.float32),
        foot_local_scaled.astype(np.float32),
    )

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
    
    scaled_root_pos, scaled_root_rot, scaled_foot_pos_world, scaled_foot_pos_local = build_scaled_targets(
        ref.root_pos, ref.root_rot, ref.foot_pos, meta, scale, args.height_offset
    )
    # NOTE: scaled_foot_pos_local is in the ORIGINAL reference root frame,
    # so the feet move relative to the (pinned) robot body exactly as they
    # moved relative to the dog's body.

    sim_cfg = sim_utils.SimulationCfg(
        dt=args.physics_dt,
        render_interval=1 if args.visualise else 4,
        device=device,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    if args.visualise:
        sim.set_camera_view(eye=[2.5, 2.0, 1.6], target=[0.0, 0.0, 0.45])
        matplotlib.use("TkAgg")

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
        print(f"[INFO] Set up foot markers for visualization")
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        foot_marker_cfg = frame_marker_cfg.replace(prim_path="/Visuals/foot_targets")
        foot_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        foot_markers = VisualizationMarkers(foot_marker_cfg)
    else:
        foot_markers = None

    retargeter = QuadrupedRetargeter(
        robot=robot, scene=scene, sim=sim, meta=meta, device=device,
        ik_iterations=args.ik_iterations, ik_damping=args.ik_damping,
        physics_dt=args.physics_dt, num_envs=num_envs,
    )

    T = ref.root_pos.shape[0]
    isaac_joint_names = list(robot.joint_names)
    isaac_default = retargeter.default_qpos[0].detach().cpu().numpy().astype(np.float32)

    # Export the FIXED base pose, broadcast across all T frames, so downstream
    # consumers see a consistent "pinned in air" trajectory matching simulation.
    out_root_pos = np.tile(
        np.array([[0.0, 0.0, 0.5]], dtype=np.float32), (T, 1)
    )
    out_root_rot = np.tile(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (T, 1)
    )
    
    out_joint_pos = np.zeros((T, len(isaac_joint_names)), dtype=np.float32)
    out_foot_pos = np.zeros((T, 4, 3), dtype=np.float32)

    skeleton_plotter = None
    if args.plot_skeleton:
        skeleton_plotter = LiveSkeletonPlotter(
            foot_pos_local_all=scaled_foot_pos_local,
            foot_names=meta["foot_bodies"],
        )

    
    print(f"[Retarget] Processing {T} frames on {device}...")
    start = time.time()
    try:
        for t in range(T):
            jpos, _ = retargeter.solve_frame(
                root_pos=out_root_pos[t],
                root_rot=out_root_rot[t],
                foot_pos_local=scaled_foot_pos_local[t],
            )
            print(f"{out_root_pos[t]}")
            out_joint_pos[t] = jpos
            robot.update(args.physics_dt)
            for leg_i, foot_id in enumerate(retargeter.foot_body_ids):
                out_foot_pos[t, leg_i] = robot.data.body_pos_w[0, foot_id].detach().cpu().numpy()

            if foot_markers is not None:
                # Targets in world frame under the pinned base: translate by
                # fixed_root_pos and rotate by identity -> just add the offset.
                world_targets = scaled_foot_pos_local[t] + np.array([0.0, 0.0, 0.5], dtype=np.float32)
                foot_markers.visualize(
                    translations=torch.tensor(world_targets, device=device, dtype=torch.float32)
                )
                sim.render()

            if skeleton_plotter is not None:
                skeleton_plotter.update(
                    root_world_t=out_root_pos[t],
                    foot_local_t=scaled_foot_pos_local[t],
                    frame_idx=t,
                    t_sec=t * ref.dt,
                )

            if (t + 1) % 25 == 0 or t == T - 1:
                elapsed = max(time.time() - start, 1e-6)
                rate = (t + 1) / elapsed
                print(f"  frame {t + 1:5d}/{T} | {rate:.1f} frames/s")
    finally:
        if skeleton_plotter is not None:
            skeleton_plotter.close()


    out_joint_vel, out_joint_acc = compute_kinematics_from_positions(out_joint_pos, ref.dt)
    joint_pos_for_training = to_training_joint_format(out_joint_pos, isaac_default)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        root_pos=out_root_pos,
        root_rot=out_root_rot,
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
