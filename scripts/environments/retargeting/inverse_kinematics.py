from __future__ import annotations
import argparse
import os
import time
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import torch
 
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

from .helpers import *


ROBOT_META = {
    "anymal_d": {
        "foot_bodies": ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],
        "leg_roots": ["LF_HIP", "RF_HIP", "LH_HIP", "RH_HIP"],
        "joint_names": [
            "LF_HAA", "LF_HFE", "LF_KFE",
            "RF_HAA", "RF_HFE", "RF_KFE",
            "LH_HAA", "LH_HFE", "LH_KFE",
            "RH_HAA", "RH_HFE", "RH_KFE",
        ],
        "default_joint_pos": [
            0.0, 0.4, -0.8,
            0.0, 0.4, -0.8,
            0.0, -0.4, 0.8,
            0.0, -0.4, 0.8,
        ],
        "leg_length": 0.55,
        "base_height": 0.55,
    }
}


def load_reference_motion(path: str, fps: float) -> ReferenceMotion:
    """Load a reference motion file and return a ReferenceMotion object.
 
    Supported raw formats
    ---------------------
    .npz / .npy : numpy archive with the keys described in ReferenceMotion
    .pkl        : pickled dict with the same keys
    .json       : JSON dict with the same keys (lists → numpy arrays)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Reference motion not found: {path}")
 
    ext = p.suffix.lower()
    if ext == ".npz":
        data = dict(np.load(path, allow_pickle=True))
    elif ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif ext == ".json":
        with open(path) as f:
            raw = json.load(f)
        data = {k: np.array(v) for k, v in raw.items()}
    else:
        raise ValueError(f"Unsupported reference motion format: {ext!r}")
 
    def _get(key, required=True):
        val = data.get(key)
        if val is None and required:
            raise KeyError(
                f"Required key '{key}' missing from reference motion file.\n"
                f"Available keys: {list(data.keys())}"
            )
        return np.asarray(val, dtype=np.float32) if val is not None else None
 
    root_pos  = _get("root_pos")           # (T, 3)
    root_rot  = _get("root_rot")           # (T, 4) wxyz
    foot_pos  = _get("foot_pos")           # (T, 4, 3)
    root_lin_vel = _get("root_lin_vel", required=False)
    root_ang_vel = _get("root_ang_vel", required=False)
 
    T = root_pos.shape[0]
    assert root_rot.shape  == (T, 4),    f"root_rot must be (T,4), got {root_rot.shape}"
    assert foot_pos.shape  == (T, 4, 3), f"foot_pos must be (T,4,3), got {foot_pos.shape}"
 
    # Compute velocities by finite-difference if not provided
    dt = 1.0 / fps
    if root_lin_vel is None:
        root_lin_vel = np.gradient(root_pos, dt, axis=0).astype(np.float32)
    if root_ang_vel is None:
        # Approximate angular velocity from quaternion derivative
        root_ang_vel = _quat_to_angular_velocity(root_rot, dt)
 
    print(
        f"[Loader] Loaded '{p.name}' — {T} frames @ {fps} Hz "
        f"({T/fps:.1f} s)."
    )
    return ReferenceMotion(
        root_pos=root_pos,
        root_rot=root_rot,
        foot_pos=foot_pos,
        root_lin_vel=root_lin_vel,
        root_ang_vel=root_ang_vel,
        dt=dt,
    )
    
    
@configclass
class RetargetSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground plane + robot."""
    ground = sim_utils.GroundPlaneCfg()
    robot: ArticulationCfg = None    # filled at runtimes
    
    
def run_retargeting(args) -> None:
    """Full pipeline: load → scene → IK solve → save."""
 
    device = args.device
    num_envs = args.num_envs
 
    # ── (a) Load reference motion ─────────────────────────────────────────────
    ref = load_reference_motion(args.ref_motion, args.fps)
    T   = ref.root_pos.shape[0]
 
    # ── (b) Retrieve robot metadata ───────────────────────────────────────────
    meta = ROBOT_META[args.robot]
 
    # ── (c) Compute animal→robot scale factors ────────────────────────────────
    scale = compute_scale_factors(ref, meta)
 
    # ── (d) Scale reference root & foot positions ─────────────────────────────
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
 
    # ── (e) Build IsaacLab simulation ─────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(
        dt=args.physics_dt,
        render_interval=1 if args.visualise else 4,
        device=device,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.5])
 
    # Scene
    robot_cfg = _get_robot_cfg(args.robot)
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
 
    # ── (f) Instantiate retargeter ────────────────────────────────────────────
    retargeter = QuadrupedRetargeter(
        robot        = robot,
        meta         = meta,
        device       = device,
        ik_iterations= args.ik_iterations,
        ik_damping   = args.ik_damping,
        physics_dt   = args.physics_dt,
        num_envs     = num_envs,
    )
 
    # ── (g) Allocate output arrays ────────────────────────────────────────────
    out_root_pos      = np.zeros((T, 3),   dtype=np.float32)
    out_root_rot      = np.zeros((T, 4),   dtype=np.float32)
    out_root_lin_vel  = np.zeros((T, 3),   dtype=np.float32)
    out_root_ang_vel  = np.zeros((T, 3),   dtype=np.float32)
    out_joint_pos     = np.zeros((T, 12),  dtype=np.float32)
    out_joint_vel     = np.zeros((T, 12),  dtype=np.float32)
    out_foot_pos      = np.zeros((T, 4, 3),dtype=np.float32)
 
    # ── (h) Frame-by-frame IK solve ───────────────────────────────────────────
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
 
    # ── (i) Compute joint accelerations by finite difference ──────────────────
    out_joint_acc = np.gradient(out_joint_vel, ref.dt, axis=0).astype(np.float32)
 
    # ── (j) Save retargeted trajectory ───────────────────────────────────────
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
        joint_acc    = out_joint_acc,     # (T, 12)
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
    _print_summary(out_path, out_joint_pos, out_joint_vel, meta)