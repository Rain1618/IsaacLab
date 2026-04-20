#!/usr/bin/env python3
"""
Animate quadruped body motion from NPZ, with side-by-side raw vs. scaled view.

Expected NPZ keys:
    - root_pos : (T, 3)
    - root_rot : (T, 4) quaternion, or (T, 3) Euler angles (roll, pitch, yaw)
    - foot_pos : (T, 4, 3)
Optional:
    - fps      : (1,)
    - dt       : (1,)

Visualization:
    - LEFT panel:  raw animal motion (root, feet, body edges, forward arrow).
    - RIGHT panel: same motion but with foot positions scaled to the target
                   robot's stance geometry (stance_length / stance_width /
                   base_height). Feet are scaled in the body frame so gait
                   structure is preserved; the root trajectory is unchanged.

Usage:
    python animate_robot_body.py --npz path/to/motion.npz \
        --stance-length 0.38 --stance-width 0.31 --base-height 0.33
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


FOOT_NAMES = [
    "front_left_paw",
    "front_right_paw",
    "back_left_paw",
    "back_right_paw",
]

BODY_EDGES = [
    ("root", "front_left_paw"),
    ("root", "front_right_paw"),
    ("root", "back_left_paw"),
    ("root", "back_right_paw"),
]


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------

def find_first_key(data, candidates, label, required=True):
    for key in candidates:
        if key in data:
            return data[key], key
    if required:
        raise KeyError(
            f"Could not find {label}. Tried keys: {candidates}. "
            f"Available keys: {list(data.keys())}"
        )
    return None, None


def load_motion_npz(npz_path: str) -> dict:
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load NPZ: {e}") from e

    root_pos, root_pos_key = find_first_key(
        data,
        ["root_pos", "root_positions", "base_pos", "base_positions"],
        "root positions", required=True,
    )
    root_rot, root_rot_key = find_first_key(
        data,
        ["root_rot", "root_orientations", "base_rot", "base_orientations", "root_quat", "base_quat"],
        "root orientations", required=False,
    )
    foot_pos, foot_pos_key = find_first_key(
        data,
        ["foot_pos", "foot_positions", "feet_pos", "feet_positions"],
        "foot positions", required=True,
    )

    fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else None
    dt = float(np.asarray(data["dt"]).reshape(-1)[0]) if "dt" in data else None

    root_pos = np.asarray(root_pos, dtype=np.float32)
    foot_pos = np.asarray(foot_pos, dtype=np.float32)
    root_rot = None if root_rot is None else np.asarray(root_rot, dtype=np.float32)

    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos shape (T, 3), got {root_pos.shape}")
    if foot_pos.ndim != 3 or foot_pos.shape[2] != 3:
        raise ValueError(f"Expected foot_pos shape (T, N, 3), got {foot_pos.shape}")
    if foot_pos.shape[0] != root_pos.shape[0]:
        raise ValueError(
            f"Frame mismatch: root_pos has {root_pos.shape[0]} frames, "
            f"foot_pos has {foot_pos.shape[0]} frames"
        )
    if foot_pos.shape[1] != 4:
        raise ValueError(f"This visualizer expects 4 feet. Got foot_pos shape {foot_pos.shape}.")

    if root_rot is not None:
        if root_rot.ndim != 2 or root_rot.shape[1] not in (3, 4):
            raise ValueError(f"Expected root_rot shape (T, 3) or (T, 4), got {root_rot.shape}")
        if root_rot.shape[0] != root_pos.shape[0]:
            raise ValueError(
                f"Frame mismatch: root_pos has {root_pos.shape[0]} frames, "
                f"root_rot has {root_rot.shape[0]} frames"
            )

    if dt is None and fps is not None and fps > 0:
        dt = 1.0 / fps
    if dt is None:
        dt = 1.0 / 30.0
    if fps is None and dt > 0:
        fps = 1.0 / dt

    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "foot_pos": foot_pos,
        "fps": fps,
        "dt": dt,
        "root_pos_key": root_pos_key,
        "root_rot_key": root_rot_key,
        "foot_pos_key": foot_pos_key,
        "available_keys": list(data.keys()),
    }


# ---------------------------------------------------------------------------
# Rotation utilities
# ---------------------------------------------------------------------------

def quat_to_rotmat(quat: np.ndarray, order: str = "xyzw") -> np.ndarray:
    """Vectorized quaternion -> rotation matrix."""
    q = np.asarray(quat, dtype=np.float64)
    if order == "xyzw":
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    elif order == "wxyz":
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    else:
        raise ValueError(f"Unknown quaternion order: {order}")

    norm = np.sqrt(x * x + y * y + z * z + w * w)
    norm = np.where(norm < 1e-8, 1.0, norm)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    T = q.shape[0]
    R = np.empty((T, 3, 3), dtype=np.float64)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)
    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[:, 1, 2] = 2.0 * (yz - wx)
    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def euler_to_rotmat(euler: np.ndarray) -> np.ndarray:
    """Vectorized Euler (roll, pitch, yaw) -> rotation matrix (ZYX intrinsic)."""
    e = np.asarray(euler, dtype=np.float64)
    r, p, y = e[:, 0], e[:, 1], e[:, 2]
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    T = e.shape[0]
    R = np.empty((T, 3, 3), dtype=np.float64)
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr
    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr
    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


def root_rot_to_rotmat(root_rot: np.ndarray, quat_order: str = "xyzw") -> np.ndarray:
    if root_rot is None:
        return None
    if root_rot.shape[1] == 4:
        return quat_to_rotmat(root_rot, order=quat_order)
    if root_rot.shape[1] == 3:
        return euler_to_rotmat(root_rot)
    raise ValueError(f"Unsupported root_rot shape: {root_rot.shape}")


# ---------------------------------------------------------------------------
# World <-> local transforms and quaternion continuity
# ---------------------------------------------------------------------------

def ensure_quaternion_continuity(root_rot: np.ndarray) -> np.ndarray:
    """Flip signs so adjacent quaternions stay in the same hemisphere:
    enforce <q_t, q_{t-1}> >= 0. No-op for (T, 3) Euler input.
    """
    if root_rot is None or root_rot.shape[1] != 4:
        return root_rot
    q = root_rot.astype(np.float64).copy()
    # Vectorized cumulative sign: s_t = prod_{k<=t} sign(<q_k, q_{k-1}>).
    dots = np.einsum("ti,ti->t", q[1:], q[:-1])     # (T-1,)
    signs = np.where(dots < 0.0, -1.0, 1.0)
    cum = np.concatenate([[1.0], np.cumprod(signs)])  # (T,)
    q = q * cum[:, None]
    return q.astype(root_rot.dtype)


def world_to_local_points(root_pos: np.ndarray,
                          root_rot: np.ndarray,
                          world_points: np.ndarray,
                          quat_order: str = "xyzw") -> np.ndarray:
    """Transform world-frame points into the per-timestep body frame.

    x_local_t = R_t^T (x_world_t - p_t)

    Shapes:
        root_pos     : (T, 3)
        root_rot     : (T, 4) quat or (T, 3) Euler
        world_points : (T, N, 3)

    Returns:
        (T, N, 3) points in body frame.
    """
    R = root_rot_to_rotmat(root_rot, quat_order=quat_order)     # (T,3,3)
    rel = world_points.astype(np.float64) - root_pos.astype(np.float64)[:, None, :]
    # Batched R^T @ rel: einsum contracts index 'j' on R's first (transpose) dim.
    local = np.einsum("tji,tnj->tni", R, rel)
    return local.astype(world_points.dtype)


def local_to_world_points(root_pos: np.ndarray,
                          root_rot: np.ndarray,
                          local_points: np.ndarray,
                          quat_order: str = "xyzw") -> np.ndarray:
    """Inverse of world_to_local_points: x_world_t = R_t x_local_t + p_t."""
    R = root_rot_to_rotmat(root_rot, quat_order=quat_order)     # (T,3,3)
    rotated = np.einsum("tij,tnj->tni", R, local_points.astype(np.float64))
    world = rotated + root_pos.astype(np.float64)[:, None, :]
    return world.astype(local_points.dtype)


# ---------------------------------------------------------------------------
# Scaling (adapted from user-provided functions; no ReferenceMotion wrapper)
# ---------------------------------------------------------------------------

def compute_scale_factors(root_pos: np.ndarray,
                          root_rot: np.ndarray,
                          foot_pos: np.ndarray,
                          meta: dict,
                          quat_order: str = "xyzw") -> dict:
    """Compute per-axis scale factors to retarget animal feet to robot stance.

    Foot ordering assumed: [FL, FR, BL, BR].
        lateral stance width      ~ mean |y_FL - y_FR|  (body frame)
        longitudinal stance length ~ mean |x_FL - x_BL| (body frame)
        base height               ~ median(z_root - mean_z_feet) (world frame)
    """
    foot_local = world_to_local_points(root_pos, root_rot, foot_pos, quat_order=quat_order)
    lat_animal = float(np.mean(np.abs(foot_local[:, 0, 1] - foot_local[:, 1, 1])))
    lon_animal = float(np.mean(np.abs(foot_local[:, 0, 0] - foot_local[:, 2, 0])))
    mean_foot_h = foot_pos[:, :, 2].mean(axis=1)
    base_h_animal = float(np.median(root_pos[:, 2] - mean_foot_h))

    robot_width = float(meta["stance_width"])
    robot_length = float(meta["stance_length"])
    robot_base_h = float(meta["base_height"])

    sx = robot_length / max(lon_animal, 1e-4)
    sy = robot_width / max(lat_animal, 1e-4)
    sz = robot_base_h / max(base_h_animal, 1e-4)

    scale = {
        "x": float(sx), "y": float(sy), "z": float(sz),
        "base_h_animal": base_h_animal,
        "robot_base_h": robot_base_h,
        "animal_stance_length": lon_animal,
        "animal_stance_width": lat_animal,
        "robot_stance_length": robot_length,
        "robot_stance_width": robot_width,
    }
    print(
        f"[Scale] animal length={lon_animal:.3f}m width={lat_animal:.3f}m "
        f"base_h={base_h_animal:.3f}m | robot length={robot_length:.3f}m "
        f"width={robot_width:.3f}m base_h={robot_base_h:.3f}m | "
        f"scale xyz=({sx:.3f}, {sy:.3f}, {sz:.3f})"
    )
    return scale


def build_scaled_targets(root_pos: np.ndarray,
                         root_rot: np.ndarray,
                         foot_pos: np.ndarray,
                         meta: dict,
                         scale: dict,
                         height_offset: float = 0.0,
                         quat_order: str = "xyzw"):
    """Returns (scaled_root_pos, root_rot, scaled_foot_pos_world, foot_local_scaled).

    Feet are scaled in the body frame of the *original* root pose so gait
    structure is preserved. The root pose itself is left unchanged (robot
    base is pinned to the reference trajectory).
    """
    root_rot = ensure_quaternion_continuity(root_rot)

    foot_local_ref = world_to_local_points(root_pos, root_rot, foot_pos, quat_order=quat_order)
    foot_local_scaled = foot_local_ref.copy()
    foot_local_scaled[..., 0] *= scale["x"]
    foot_local_scaled[..., 1] *= scale["y"]
    foot_local_scaled[..., 2] *= scale["z"]

    scaled_root_pos = root_pos.astype(np.float32).copy()
    scaled_foot_pos_world = local_to_world_points(
        scaled_root_pos, root_rot, foot_local_scaled, quat_order=quat_order
    )
    return (
        scaled_root_pos.astype(np.float32),
        root_rot.astype(np.float32),
        scaled_foot_pos_world.astype(np.float32),
        foot_local_scaled.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def get_global_metrics(root_pos: np.ndarray, foot_pos: np.ndarray) -> dict:
    all_pts = np.concatenate([root_pos[:, None, :], foot_pos], axis=1).reshape(-1, 3)
    finite_mask = np.isfinite(all_pts).all(axis=1)
    if not np.any(finite_mask):
        raise ValueError("No finite points found in motion data.")
    all_pts = all_pts[finite_mask]

    mins = np.min(all_pts, axis=0)
    maxs = np.max(all_pts, axis=0)
    center = 0.5 * (mins + maxs)
    half_range = 0.5 * np.max(maxs - mins)
    if half_range < 1e-6:
        half_range = 1.0
    half_range += 0.1 * half_range

    return {
        "x_min": float(mins[0]), "x_max": float(maxs[0]),
        "y_min": float(mins[1]), "y_max": float(maxs[1]),
        "z_min": float(mins[2]), "z_max": float(maxs[2]),
        "bounds": {
            "x": (float(center[0] - half_range), float(center[0] + half_range)),
            "y": (float(center[1] - half_range), float(center[1] + half_range)),
            "z": (float(center[2] - half_range), float(center[2] + half_range)),
        },
        "half_range": float(half_range),
    }


def apply_axis_flips(points, metrics, flip_x, flip_y, flip_z):
    pts = points.copy()
    if flip_x: pts[:, 0] = (metrics["x_max"] + metrics["x_min"]) - pts[:, 0]
    if flip_y: pts[:, 1] = (metrics["y_max"] + metrics["y_min"]) - pts[:, 1]
    if flip_z: pts[:, 2] = -pts[:, 2]
    return pts


def apply_axis_flips_to_vector(vec, flip_x, flip_y, flip_z):
    v = vec.copy()
    if flip_x: v[..., 0] = -v[..., 0]
    if flip_y: v[..., 1] = -v[..., 1]
    if flip_z: v[..., 2] = -v[..., 2]
    return v


def build_frame_points(root_t, feet_t):
    return {
        "root": root_t,
        "front_left_paw": feet_t[0],
        "front_right_paw": feet_t[1],
        "back_left_paw": feet_t[2],
        "back_right_paw": feet_t[3],
    }


def apply_axes_formatting(ax, metrics):
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_xlim(metrics["bounds"]["x"])
    ax.set_ylim(metrics["bounds"]["y"])
    ax.set_zlim(metrics["bounds"]["z"])
    try:
        xr = metrics["bounds"]["x"][1] - metrics["bounds"]["x"][0]
        yr = metrics["bounds"]["y"][1] - metrics["bounds"]["y"][0]
        zr = metrics["bounds"]["z"][1] - metrics["bounds"]["z"][0]
        ax.set_box_aspect((xr, yr, zr))
    except Exception:
        pass


def draw_panel(ax, frame_idx, root_pos, foot_pos, rotmats, arrow_len, metrics, args, fps, title_prefix):
    """Render one 3D panel (shared between raw and scaled views)."""
    ax.clear()

    root_t = root_pos[frame_idx]
    feet_t = foot_pos[frame_idx]

    pts_dict = build_frame_points(root_t, feet_t)
    names = list(pts_dict.keys())

    pts = np.stack([pts_dict[name] for name in names], axis=0)
    pts = apply_axis_flips(pts, metrics, args.flip_x, args.flip_y, args.flip_z)
    pts_dict = {name: pts[i] for i, name in enumerate(names)}

    root_draw = pts_dict["root"]
    feet_draw = np.stack([pts_dict[name] for name in FOOT_NAMES], axis=0)

    ax.scatter(root_draw[0], root_draw[1], root_draw[2], s=90, label="root")
    ax.scatter(feet_draw[:, 0], feet_draw[:, 1], feet_draw[:, 2], s=50, label="feet")

    for a, b in BODY_EDGES:
        pa, pb = pts_dict[a], pts_dict[b]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], linewidth=2)

    if rotmats is not None:
        R_t = rotmats[frame_idx]
        fwd = apply_axis_flips_to_vector(R_t[:, 0], args.flip_x, args.flip_y, args.flip_z)
        ax.quiver(root_draw[0], root_draw[1], root_draw[2],
                  fwd[0], fwd[1], fwd[2],
                  length=arrow_len, normalize=False,
                  color="red", linewidth=2.5, arrow_length_ratio=0.25)
        if args.show_body_frame:
            left = apply_axis_flips_to_vector(R_t[:, 1], args.flip_x, args.flip_y, args.flip_z)
            up = apply_axis_flips_to_vector(R_t[:, 2], args.flip_x, args.flip_y, args.flip_z)
            ax.quiver(root_draw[0], root_draw[1], root_draw[2], left[0], left[1], left[2],
                      length=arrow_len, normalize=False, color="green",
                      linewidth=2.0, arrow_length_ratio=0.25)
            ax.quiver(root_draw[0], root_draw[1], root_draw[2], up[0], up[1], up[2],
                      length=arrow_len, normalize=False, color="blue",
                      linewidth=2.0, arrow_length_ratio=0.25)

    if args.show_trail:
        start = max(0, frame_idx - args.trail_length)
        trail = root_pos[start:frame_idx + 1]
        trail = apply_axis_flips(trail, metrics, args.flip_x, args.flip_y, args.flip_z)
        ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], linewidth=1)

    if args.show_labels:
        for name, p in pts_dict.items():
            ax.text(p[0], p[1], p[2], name, fontsize=7)

    time_sec = frame_idx / fps
    ax.set_title(f"{title_prefix} | frame={frame_idx} | t={time_sec:.2f}s", fontsize=10)
    apply_axes_formatting(ax, metrics)


def update(frame_idx,
           raw_root, raw_feet, raw_rotmats, raw_metrics, raw_arrow_len,
           scl_root, scl_feet, scl_rotmats, scl_metrics, scl_arrow_len,
           args, ax_raw, ax_scl, fps):
    draw_panel(ax_raw, frame_idx, raw_root, raw_feet, raw_rotmats,
               raw_arrow_len, raw_metrics, args, fps, title_prefix="Raw animal motion")
    draw_panel(ax_scl, frame_idx, scl_root, scl_feet, scl_rotmats,
               scl_arrow_len, scl_metrics, args, fps, title_prefix="Scaled to robot stance (normalized [-1, 1])")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Animate quadruped body motion from NPZ (raw + scaled).")
    parser.add_argument("--npz", required=True, help="Path to motion .npz file")
    parser.add_argument("--interval", type=int, default=33, help="Animation delay in ms")
    parser.add_argument("--flip-x", action="store_true", help="Mirror x axis")
    parser.add_argument("--flip-y", action="store_true", default=False, help="Mirror y axis")
    parser.add_argument("--flip-z", action="store_true", help="Invert z axis")
    parser.add_argument("--show-labels", action="store_true", help="Show point labels")
    parser.add_argument("--show-trail", action="store_true", help="Show trailing root path")
    parser.add_argument("--trail-length", type=int, default=20, help="Trail length in frames")
    parser.add_argument("--elev", type=float, default=20.0, help="Camera elevation")
    parser.add_argument("--azim", type=float, default=-100.0, help="Camera azimuth")
    parser.add_argument("--quat-order", choices=["xyzw", "wxyz"], default="xyzw",
                        help="Quaternion layout when root_rot has shape (T, 4).")
    parser.add_argument("--show-body-frame", action="store_true",
                        help="Draw full body triad (forward/left/up) instead of just forward.")
    parser.add_argument("--arrow-scale", type=float, default=0.15,
                        help="Arrow length as fraction of scene half-range.")

    # Target robot dimensions (defaults loosely match Unitree Go1/Go2).
    parser.add_argument("--stance-length", type=float, default=0.38,
                        help="Target robot longitudinal stance length in meters (FL-BL distance).")
    parser.add_argument("--stance-width", type=float, default=0.31,
                        help="Target robot lateral stance width in meters (FL-FR distance).")
    parser.add_argument("--base-height", type=float, default=0.33,
                        help="Target robot nominal base height in meters.")
    parser.add_argument("--height-offset", type=float, default=0.0,
                        help="Optional vertical offset added to scaled root.")
    args = parser.parse_args()

    try:
        data = load_motion_npz(args.npz)
    except Exception as e:
        print(str(e), file=sys.stderr); sys.exit(1)

    root_pos = data["root_pos"]
    foot_pos = data["foot_pos"]
    root_rot = data["root_rot"]
    fps = data["fps"]

    if root_rot is None:
        print("ERROR: root_rot is required for scaled retargeting but was not found.",
              file=sys.stderr)
        sys.exit(1)

    # Rotation matrices for the raw (unscaled) motion (used for arrows).
    raw_rotmats = root_rot_to_rotmat(root_rot, quat_order=args.quat_order)

    # Compute per-axis scale factors and build the scaled trajectory.
    meta = {
        "stance_length": args.stance_length,
        "stance_width": args.stance_width,
        "base_height": args.base_height,
    }
    scale = compute_scale_factors(root_pos, root_rot, foot_pos, meta,
                                  quat_order=args.quat_order)

    scl_root, scl_rot, scl_feet, _scl_foot_local = build_scaled_targets(
        root_pos, root_rot, foot_pos, meta, scale,
        height_offset=args.height_offset, quat_order=args.quat_order,
    )
    scl_rotmats = root_rot_to_rotmat(scl_rot, quat_order=args.quat_order)

    print("Loaded NPZ successfully.")
    print(f"  root positions : {data['root_pos_key']}  shape={root_pos.shape}")
    print(f"  foot positions : {data['foot_pos_key']}  shape={foot_pos.shape}")
    rep = "quaternion" if root_rot.shape[1] == 4 else "euler"
    print(f"  root orient.   : {data['root_rot_key']}  shape={root_rot.shape}  ({rep})")
    print(f"  dt={data['dt']:.6f}, fps={data['fps']:.3f}")

    # Raw panel: data-driven bounds. Scaled panel: fixed [-1, 1] cube so the
    # retargeted motion is shown in a normalized viewing frame (points outside
    # the cube will clip at the axis limits by design).
    raw_metrics = get_global_metrics(root_pos, foot_pos)
    scl_metrics = {
        "x_min": -1.0, "x_max": 1.0,
        "y_min": -1.0, "y_max": 1.0,
        "z_min": -1.0, "z_max": 1.0,
        "bounds": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0)},
        "half_range": 1.0,
    }
    raw_arrow_len = args.arrow_scale * raw_metrics["half_range"]
    scl_arrow_len = args.arrow_scale * scl_metrics["half_range"]

    fig = plt.figure(figsize=(14, 7))
    ax_raw = fig.add_subplot(1, 2, 1, projection="3d")
    ax_scl = fig.add_subplot(1, 2, 2, projection="3d")
    for ax in (ax_raw, ax_scl):
        ax.view_init(elev=args.elev, azim=args.azim)

    fig.suptitle(
        f"Raw animal vs. robot-scaled feet  |  "
        f"scale (x,y,z) = ({scale['x']:.2f}, {scale['y']:.2f}, {scale['z']:.2f})",
        fontsize=12,
    )

    scl_root_pos_local = np.zeros_like(scl_root)

    ani = FuncAnimation(
        fig, update,
        frames=root_pos.shape[0],
        fargs=(root_pos, foot_pos, raw_rotmats, raw_metrics, raw_arrow_len,
               scl_root_pos_local, _scl_foot_local, scl_rotmats, scl_metrics, scl_arrow_len,
               args, ax_raw, ax_scl, fps),
        interval=args.interval, blit=False, repeat=True,
    )
    _ = ani
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()