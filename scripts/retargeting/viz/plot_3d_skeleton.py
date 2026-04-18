#!/usr/bin/env python3
"""
Animate quadruped body motion from NPZ.

Expected NPZ keys:
    - root_pos : (T, 3)
    - root_rot : (T, 4) or (T, 3)   [loaded but not required for drawing]
    - foot_pos : (T, 4, 3)
Optional:
    - fps      : (1,)
    - dt       : (1,)

Visualization:
    - root point
    - four foot points
    - lines from root to each foot
    - optional root trajectory trail

Usage:
    python animate_robot_body.py --npz path/to/motion.npz

Optional:
    python animate_robot_body.py --npz path/to/motion.npz --flip-z --show-labels --show-trail
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
        "root positions",
        required=True,
    )
    root_rot, root_rot_key = find_first_key(
        data,
        ["root_rot", "root_orientations", "base_rot", "base_orientations", "root_quat", "base_quat"],
        "root orientations",
        required=False,
    )
    foot_pos, foot_pos_key = find_first_key(
        data,
        ["foot_pos", "foot_positions", "feet_pos", "feet_positions"],
        "foot positions",
        required=True,
    )

    fps = None
    dt = None

    if "fps" in data:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if "dt" in data:
        dt = float(np.asarray(data["dt"]).reshape(-1)[0])

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
        raise ValueError(
            f"This visualizer expects 4 feet. Got foot_pos shape {foot_pos.shape}."
        )

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

    pad = 0.1 * half_range
    half_range += pad

    return {
        "x_min": float(mins[0]),
        "x_max": float(maxs[0]),
        "y_min": float(mins[1]),
        "y_max": float(maxs[1]),
        "z_min": float(mins[2]),
        "z_max": float(maxs[2]),
        "bounds": {
            "x": (float(center[0] - half_range), float(center[0] + half_range)),
            "y": (float(center[1] - half_range), float(center[1] + half_range)),
            "z": (float(center[2] - half_range), float(center[2] + half_range)),
        },
    }


def apply_axis_flips(points: np.ndarray, metrics: dict, flip_x: bool, flip_y: bool, flip_z: bool) -> np.ndarray:
    pts = points.copy()

    if flip_x:
        pts[:, 0] = (metrics["x_max"] + metrics["x_min"]) - pts[:, 0]
    if flip_y:
        pts[:, 1] = (metrics["y_max"] + metrics["y_min"]) - pts[:, 1]
    if flip_z:
        pts[:, 2] = -pts[:, 2]

    return pts


def build_frame_points(root_t: np.ndarray, feet_t: np.ndarray) -> dict:
    return {
        "root": root_t,
        "front_left_paw": feet_t[0],
        "front_right_paw": feet_t[1],
        "back_left_paw": feet_t[2],
        "back_right_paw": feet_t[3],
    }


def apply_axes_formatting(ax, metrics: dict):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
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


def update(frame_idx, root_pos, foot_pos, metrics, args, ax, fps):
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

    # Scatter root and feet
    ax.scatter(root_draw[0], root_draw[1], root_draw[2], s=90, label="root")
    ax.scatter(feet_draw[:, 0], feet_draw[:, 1], feet_draw[:, 2], s=50, label="feet")

    # Draw body edges
    for a, b in BODY_EDGES:
        pa = pts_dict[a]
        pb = pts_dict[b]
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            [pa[2], pb[2]],
            linewidth=2,
        )

    # Optional root trail
    if args.show_trail:
        start = max(0, frame_idx - args.trail_length)
        trail = root_pos[start:frame_idx + 1]
        trail = apply_axis_flips(trail, metrics, args.flip_x, args.flip_y, args.flip_z)
        ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], linewidth=1)

    # Optional labels
    if args.show_labels:
        for name, p in pts_dict.items():
            ax.text(p[0], p[1], p[2], name, fontsize=8)

    time_sec = frame_idx / fps
    ax.set_title(f"Robot body animation | frame={frame_idx} | t={time_sec:.2f}s")
    apply_axes_formatting(ax, metrics)


def main():
    parser = argparse.ArgumentParser(description="Animate quadruped body motion from NPZ.")
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
    args = parser.parse_args()

    try:
        data = load_motion_npz(args.npz)
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    root_pos = data["root_pos"]
    foot_pos = data["foot_pos"]
    fps = data["fps"]

    print("Loaded NPZ successfully.")
    print(f"  root positions key    : {data['root_pos_key']}  shape={root_pos.shape}")
    print(f"  foot positions key    : {data['foot_pos_key']}  shape={foot_pos.shape}")
    if data["root_rot_key"] is not None:
        print(f"  root orientations key : {data['root_rot_key']}  shape={data['root_rot'].shape}")
    print(f"  dt={data['dt']:.6f}, fps={data['fps']:.3f}")
    print(f"  available keys: {data['available_keys']}")

    try:
        metrics = get_global_metrics(root_pos, foot_pos)
    except Exception as e:
        print(f"Failed to compute plot bounds: {e}", file=sys.stderr)
        sys.exit(1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)

    ani = FuncAnimation(
        fig,
        update,
        frames=root_pos.shape[0],
        fargs=(root_pos, foot_pos, metrics, args, ax, fps),
        interval=args.interval,
        blit=False,
        repeat=True,
    )

    # Prevent animation object from being garbage collected
    _ = ani

    plt.show()


if __name__ == "__main__":
    main()