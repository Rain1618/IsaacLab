
import argparse

import time 
import torch

import numpy as np
from pathlib import Path

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.controllers import (  # noqa: E402
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    quat_apply,
    quat_conjugate,
    quat_from_euler_xyz,
    quat_mul,
    subtract_frame_transforms,
)

from .helpers import ROBOT_META, get_robot_cfg, compute_scale_factors, load_reference_motion, RetargetSceneCfg, QuadrupedRetargeter

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--ref-motion", 
        type=str,
        required=True
    )
    p.add_argument(
        "--output", 
        type=str,
        required=True
    )
    p.add_argument(
        "--physics_dt",
        type=float,
        default=1.0 / 200.0
    )
    p.add_argument(
        "--ik_damping",
        type=float,
        default=0.05
    )
    
    # IsaacLab simulation arguments
    p.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel simulation environments (default: 1).",
    )
    p.add_argument("--headless", action="store_true", default=False)
    p.add_argument("--device", type=str, default="cuda:0")
    
    return p

# -- Parse early so IsaacLab Launcher sees the sys.argv
parser = _build_arg_parser()

from isaaclab.app import AppLauncher  # noqa: E402  (import after argparse setup)
 
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
 
# Launch the simulation application (headless unless --visualise requested)
args.headless = not args.visualise
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

@configclass
class RetargetSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground plane + robot."""
    ground = sim_utils.GroundPlaneCfg()
    robot: ArticulationCfg = None    # filled at runtime

# Main function to run that invokes the Isaac AppLauncher
def run_targeting(args) -> None:
    """Main execution function which starts AppLauncher and performs motion retargeting
    
    Keyword arguments:
    args -- arguments from the CLI execution runtime, TODO: add schema for the dict obj
    Return: None
    """
    device = args.device
    num_envs = args.num_envs
 
    # Load reference motion
    ref = load_reference_motion(args.ref_motion, args.fps)
    T   = ref.root_pos.shape[0]
 
    # Retrieve robot metadata
    meta = ROBOT_META[args.robot]
 
    # Compute animal-to-robot scale factors
    scale = compute_scale_factors(ref, meta)
 
    # Scale reference root & foot positions
    scaled_root_pos = ref.root_pos.copy()
    scaled_root_pos[:, 2] = (
        (ref.root_pos[:, 2] - ref.root_pos[:, 2].min())
        * scale["z"]
        + meta["base_height"]
        + args.height_offset
    )
 
    # Scale foot XY relative to the (scaled) base
    scaled_foot_pos = ref.foot_pos.copy()
    for leg_i in range(4):
        scaled_foot_pos[:, leg_i, 2] = np.clip(
            ref.foot_pos[:, leg_i, 2] * scale["z"],
            -meta["leg_length"],
            0.05,
        )
 
    # Build IsaacLab simulation
    sim_cfg = sim_utils.SimulationCfg(
        dt=args.physics_dt,
        render_interval=1 if args.visualise else 4,
        device=device,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.5])
 
    # Scene
    robot_cfg = get_robot_cfg(args.robot)
    scene_cfg = RetargetSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene_cfg.robot = robot_cfg
    scene = InteractiveScene(scene_cfg)
 
    # Optionally: add frame markers for each foot target
    if args.visualise:
        marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/foot_targets")
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        foot_markers = VisualizationMarkers(marker_cfg)
 
    sim.reset()
    robot: Articulation = scene["robot"]
 
    # Instantiate retargeter
    retargeter = QuadrupedRetargeter(
        robot        = robot,
        meta         = meta,
        device       = device,
        ik_iterations= args.ik_iterations,
        ik_damping   = args.ik_damping,
        physics_dt   = args.physics_dt,
        num_envs     = num_envs,
    )
 
    # Allocate output arrays
    out_root_pos      = np.zeros((T, 3),   dtype=np.float32)
    out_root_rot      = np.zeros((T, 4),   dtype=np.float32)
    out_root_lin_vel  = np.zeros((T, 3),   dtype=np.float32)
    out_root_ang_vel  = np.zeros((T, 3),   dtype=np.float32)
    out_joint_pos     = np.zeros((T, 12),  dtype=np.float32)
    out_joint_vel     = np.zeros((T, 12),  dtype=np.float32)
    out_foot_pos      = np.zeros((T, 4, 3),dtype=np.float32)
 
    # Frame-by-frame IK solve
    print(f"\n[Retarget] Processing {T} frames on {device} ...")
    t_start = time.time()
 
    for t in range(T):
        root_pos_t = scaled_root_pos[t]        # (3,)
        root_rot_t = ref.root_rot[t]           # (4,) wxyz
        foot_pos_t = scaled_foot_pos[t]        # (4, 3)
 
        jpos, jvel = retargeter.solve_frame(root_pos_t, root_rot_t, foot_pos_t)
 
        # Store outputs
        out_root_pos[t]     = root_pos_t
        out_root_rot[t]     = root_rot_t
        out_root_lin_vel[t] = ref.root_lin_vel[t] * scale["z"]   # scale velocity
        out_root_ang_vel[t] = ref.root_ang_vel[t]
        out_joint_pos[t]    = jpos
        out_joint_vel[t]    = jvel
 
        # Re-read actual foot positions from sim for ground truth
        robot.update(args.physics_dt)
        for leg_i in range(4):
            fid = retargeter.foot_body_ids[leg_i]
            out_foot_pos[t, leg_i] = (
                robot.data.body_pos_w[0, fid].cpu().numpy()
            )
 
        # Visualise foot markers
        if args.visualise:
            marker_pos = torch.tensor(
                foot_pos_t, dtype=torch.float32, device=device
            ).unsqueeze(0).reshape(1, 4, 3)
            # Expand for num_envs if needed
            foot_markers.visualize(
                translations=marker_pos.reshape(-1, 3)
            )
            sim.render()
 
        # Progress
        if (t + 1) % 50 == 0 or t == T - 1:
            elapsed = time.time() - t_start
            fps_actual = (t + 1) / elapsed
            eta = (T - t - 1) / max(fps_actual, 1e-3)
            print(
                f"  Frame {t+1:5d}/{T}  |  "
                f"{fps_actual:.1f} frames/s  |  "
                f"ETA {eta:.1f}s"
            )
 
    elapsed_total = time.time() - t_start
    print(f"\n[Retarget] Done — {T} frames in {elapsed_total:.1f}s "
          f"({T/elapsed_total:.1f} frames/s).")
 
    # Compute joint accelerations by finite difference
    out_joint_acc = np.gradient(out_joint_vel, ref.dt, axis=0).astype(np.float32)
 
    # Save retargeted trajectory 
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        # Root state
        root_pos     = out_root_pos,      # (T, 3)
        root_rot     = out_root_rot,      # (T, 4)  wxyz
        root_lin_vel = out_root_lin_vel,  # (T, 3)
        root_ang_vel = out_root_ang_vel,  # (T, 3)
        # Joint state
        joint_pos    = out_joint_pos,     # (T, 12)
        joint_vel    = out_joint_vel,     # (T, 12)
        # Foot positions (in world frame)
        foot_pos     = out_foot_pos,      # (T, 4, 3)
        # Metadata
        fps          = np.array([args.fps]),
        robot        = np.array([args.robot]),
        joint_names  = np.array(meta["joint_names"]),
        foot_names   = np.array(meta["foot_bodies"]),
        dt           = np.array([ref.dt]),
    )
 
    print(f"[Retarget] Saved retargeted trajectory → '{out_path}'")    
    
    
if __name__ == "__main__":
    try:
        run_targeting(args)
    finally:
        simulation_app.close()