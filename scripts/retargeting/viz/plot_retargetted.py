#!/usr/bin/env python3
"""
Animate retargeted quadruped motion produced by ``retarget.py``.

Expected NPZ keys (as saved by retarget.py):
    - root_pos              : (T, 3)    per-frame root translation (world)
    - root_rot              : (T, 4)    per-frame root orientation (wxyz, world<-body)
    - joint_pos_isaac       : (T, 12)   retargeted joint angles (Isaac order)
    - joint_vel             : (T, 12)
    - joint_acc             : (T, 12)
    - foot_pos              : (T, 4, 3) achieved foot world positions
    - foot_pos_local        : (T, 4, 3) IK foot targets in the base frame
    - joint_names           : (12,)     e.g. ["LF_HAA", "LF_HFE", ...]
    - foot_names            : (4,)      e.g. ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
    - fps, dt, frame_duration

This script visualises the retargeted motion *without* assuming the root is
pinned. Concretely, `foot_pos_local` lives in the (moving) body frame, so we
express achieved feet in the same body frame at every frame via

    p_body(t) = R(q_root(t))^{-1} (p_world(t) - p_root(t))

where R(q) is the rotation encoded by quaternion q (wxyz). The skeleton panel
then consistently shows root-at-origin with target/achieved feet overlaid in
the base frame, so the residual arrows remain meaningful even when the root
translates and rotates.

Layout:
    LEFT panel  : 3D skeleton in the per-frame body frame. Root at origin;
                  four edges from root to each foot. Two sets of feet are
                  drawn: IK targets (filled) and IK achieved (open, in the
                  same body frame via the per-frame rotation). The view is
                  rotatable with the mouse.

    RIGHT panel : Joint-angle time series for all 12 joints, grouped by leg
                  (LF, RF, LH, RH), with a vertical cursor at the current
                  frame.

Usage:
    python animate_retarget_output.py --npz path/to/retargeted.npz
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------

REQUIRED_KEYS = [
    "root_pos", "root_rot",
    "joint_pos_isaac", "joint_names",
    "foot_pos", "foot_pos_local", "foot_names",
]


def load_retarget_npz(npz_path: str) -> dict:
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load NPZ: {e}") from e

    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise KeyError(
            f"NPZ at {npz_path!r} is missing required keys: {missing}. "
            f"Available keys: {list(data.keys())}"
        )

    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(data["root_rot"], dtype=np.float32)
    joint_pos = np.asarray(data["joint_pos_isaac"], dtype=np.float32)
    foot_pos = np.asarray(data["foot_pos"], dtype=np.float32)
    foot_pos_local = np.asarray(data["foot_pos_local"], dtype=np.float32)
    joint_names = [str(s) for s in np.asarray(data["joint_names"]).tolist()]
    foot_names = [str(s) for s in np.asarray(data["foot_names"]).tolist()]

    # fps / dt handling: retarget.py stores both.
    if "fps" in data:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    else:
        fps = None
    if "dt" in data:
        dt = float(np.asarray(data["dt"]).reshape(-1)[0])
    elif "frame_duration" in data:
        dt = float(np.asarray(data["frame_duration"]).reshape(-1)[0])
    else:
        dt = None
    if dt is None and fps is not None and fps > 0:
        dt = 1.0 / fps
    if fps is None and dt is not None and dt > 0:
        fps = 1.0 / dt
    if fps is None:
        fps, dt = 30.0, 1.0 / 30.0

    # Shape sanity
    T = root_pos.shape[0]
    if root_pos.shape != (T, 3):
        raise ValueError(f"Expected root_pos (T, 3), got {root_pos.shape}")
    if root_rot.shape != (T, 4):
        raise ValueError(f"Expected root_rot (T, 4), got {root_rot.shape}")
    if foot_pos.shape != (T, 4, 3):
        raise ValueError(f"Expected foot_pos (T, 4, 3), got {foot_pos.shape}")
    if foot_pos_local.shape != (T, 4, 3):
        raise ValueError(f"Expected foot_pos_local (T, 4, 3), got {foot_pos_local.shape}")
    if joint_pos.shape != (T, len(joint_names)):
        raise ValueError(
            f"joint_pos_isaac shape {joint_pos.shape} does not match "
            f"(T={T}, num_joints={len(joint_names)})."
        )
    if len(foot_names) != 4:
        raise ValueError(f"Expected 4 foot names, got {len(foot_names)}.")

    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "joint_pos": joint_pos,
        "joint_names": joint_names,
        "foot_pos": foot_pos,
        "foot_pos_local": foot_pos_local,
        "foot_names": foot_names,
        "fps": fps,
        "dt": dt,
        "available_keys": list(data.keys()),
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalise quaternions along the last axis. Shape preserved."""
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(n, 1e-8, None)


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate batched 3-vectors by batched wxyz quaternions.

    Uses the closed-form Rodrigues expansion
        v' = v + 2 w (u x v) + 2 (u x (u x v)),
    which avoids building 3x3 rotation matrices. Broadcasts over the
    leading dims of q and v.

    Args:
        q: (..., 4) unit wxyz quaternions.
        v: (..., 3) vectors.

    Returns:
        (..., 3) rotated vectors.
    """
    q = _normalize_quat(q)
    w = q[..., 0:1]
    u = q[..., 1:4]
    uxv = np.cross(u, v)
    uxuxv = np.cross(u, uxv)
    return (v + 2.0 * w * uxv + 2.0 * uxuxv).astype(np.float32)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of a wxyz quaternion: negate the vector part."""
    out = np.asarray(q, dtype=np.float32).copy()
    out[..., 1:] *= -1.0
    return out


def world_to_body_points(pts_world: np.ndarray,
                         root_pos: np.ndarray,
                         root_rot: np.ndarray) -> np.ndarray:
    """Transform world-frame points into the per-frame body frame.

    Args:
        pts_world: (T, K, 3) world-frame points (K keypoints per frame).
        root_pos:  (T, 3)    root positions in world.
        root_rot:  (T, 4)    root orientations as wxyz (world<-body).

    Returns:
        (T, K, 3) body-frame points at each frame.
    """
    rel = pts_world - root_pos[:, None, :]                    # (T, K, 3)
    q_inv = quat_conjugate(root_rot)                          # (T, 4)
    K = rel.shape[1]
    q_inv_b = np.broadcast_to(q_inv[:, None, :], (rel.shape[0], K, 4))
    return quat_rotate(q_inv_b, rel)


def compute_bounds(points_list, pad_frac: float = 0.10) -> dict:
    """Axis-equal bounding cube across a list of (..., 3) arrays."""
    all_pts = np.concatenate([p.reshape(-1, 3) for p in points_list], axis=0)
    finite = np.isfinite(all_pts).all(axis=1)
    if not np.any(finite):
        raise ValueError("No finite points found in skeleton data.")
    all_pts = all_pts[finite]
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    half = 0.5 * float(np.max(maxs - mins))
    half = max(half, 1e-3) * (1.0 + pad_frac)
    return {
        "x": (float(center[0] - half), float(center[0] + half)),
        "y": (float(center[1] - half), float(center[1] + half)),
        "z": (float(center[2] - half), float(center[2] + half)),
        "half_range": half,
    }


# ---------------------------------------------------------------------------
# Joint grouping
# ---------------------------------------------------------------------------

# retarget.py / helpers.ROBOT_META canonical leg order.
LEG_PREFIXES = ["LF", "RF", "LH", "RH"]
JOINT_SUFFIXES = ["HAA", "HFE", "KFE"]


def build_joint_layout(joint_names):
    """Return a (4, 3) array of joint indices grouped as
    rows = legs (LF, RF, LH, RH), cols = joints (HAA, HFE, KFE).

    Falls back gracefully if names don't match the expected convention.
    """
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    layout = np.full((len(LEG_PREFIXES), len(JOINT_SUFFIXES)), -1, dtype=np.int64)
    for li, leg in enumerate(LEG_PREFIXES):
        for ji, suf in enumerate(JOINT_SUFFIXES):
            key = f"{leg}_{suf}"
            if key in name_to_idx:
                layout[li, ji] = name_to_idx[key]
    if (layout < 0).any():
        print("[warn] Joint names do not match LF/RF/LH/RH × HAA/HFE/KFE; "
              "falling back to linear grouping.")
        linear = np.arange(len(joint_names))
        padded = np.full(12, -1, dtype=np.int64)
        padded[:min(12, len(linear))] = linear[:12]
        layout = padded.reshape(4, 3)
    return layout


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_skeleton_panel(ax, frame_idx, foot_target_local,
                        foot_achieved_local, foot_names, bounds, args, fps,
                        root_pos=None, root_rot=None):
    """3D skeleton panel in the per-frame body frame.

    Root sits at the origin of its own body frame at every frame, so we do
    not need to plot root_pos here — but we surface it (plus the root
    orientation) in the title so users can confirm that the root is no
    longer pinned.

    Preserves the current camera angles (which the user may have changed by
    dragging with the mouse) across the ax.clear() call that redraws the
    frame.
    """
    try:
        cur_elev = float(ax.elev)
        cur_azim = float(ax.azim)
    except Exception:
        cur_elev, cur_azim = args.elev, args.azim

    ax.clear()
    ax.view_init(elev=cur_elev, azim=cur_azim)

    root = np.zeros(3, dtype=np.float32)      # body-frame origin
    f_tgt = foot_target_local[frame_idx]      # (4, 3) body frame
    f_ach = foot_achieved_local[frame_idx]    # (4, 3) body frame (via R^{-1})

    # Root
    ax.scatter(root[0], root[1], root[2], s=90, color="black", label="root")

    # Feet: filled = target, open = achieved.
    ax.scatter(f_tgt[:, 0], f_tgt[:, 1], f_tgt[:, 2],
               s=60, marker="o", color="tab:blue", label="foot target")
    ax.scatter(f_ach[:, 0], f_ach[:, 1], f_ach[:, 2],
               s=60, marker="o", facecolors="none",
               edgecolors="tab:orange", linewidths=1.5, label="foot achieved")

    # Edges: root -> target (solid) and root -> achieved (dashed thin),
    # plus a short residual line between target and achieved.
    for i in range(4):
        ax.plot([root[0], f_tgt[i, 0]],
                [root[1], f_tgt[i, 1]],
                [root[2], f_tgt[i, 2]],
                linewidth=2.0, color="tab:blue", alpha=0.9)
        ax.plot([root[0], f_ach[i, 0]],
                [root[1], f_ach[i, 1]],
                [root[2], f_ach[i, 2]],
                linewidth=1.0, color="tab:orange", alpha=0.6, linestyle="--")
        ax.plot([f_tgt[i, 0], f_ach[i, 0]],
                [f_tgt[i, 1], f_ach[i, 1]],
                [f_tgt[i, 2], f_ach[i, 2]],
                linewidth=1.5, color="red", alpha=0.7)

    if args.show_labels:
        for i, name in enumerate(foot_names):
            ax.text(f_tgt[i, 0], f_tgt[i, 1], f_tgt[i, 2], name, fontsize=7)

    # Mean residual in meters for a quick diagnostic in the title.
    resid = np.linalg.norm(f_tgt - f_ach, axis=1)
    mean_resid = float(np.mean(resid))
    max_resid = float(np.max(resid))

    ax.set_xlabel("x (body)"); ax.set_ylabel("y (body)"); ax.set_zlabel("z (body)")
    ax.set_xlim(bounds["x"]); ax.set_ylim(bounds["y"]); ax.set_zlim(bounds["z"])
    try:
        xr = bounds["x"][1] - bounds["x"][0]
        yr = bounds["y"][1] - bounds["y"][0]
        zr = bounds["z"][1] - bounds["z"][0]
        ax.set_box_aspect((xr, yr, zr))
    except Exception:
        pass

    t_sec = frame_idx / fps
    # Report the (no-longer-pinned) root pose in the title so it is visible
    # that this script is now using the per-frame root_pos / root_rot.
    if root_pos is not None and root_rot is not None:
        rp = root_pos[frame_idx]
        rr = root_rot[frame_idx]
        root_line = (f"root_pos=({rp[0]:+.2f}, {rp[1]:+.2f}, {rp[2]:+.2f})  "
                     f"root_rot(wxyz)=({rr[0]:+.2f}, {rr[1]:+.2f}, "
                     f"{rr[2]:+.2f}, {rr[3]:+.2f})")
    else:
        root_line = ""

    ax.set_title(
        f"Skeleton in body frame | frame={frame_idx} | t={t_sec:.2f}s\n"
        f"residual target→achieved: mean={mean_resid*1000:.1f}mm, "
        f"max={max_resid*1000:.1f}mm"
        + (f"\n{root_line}" if root_line else ""),
        fontsize=9,
    )
    if frame_idx == 0 and args.show_legend:
        ax.legend(loc="upper right", fontsize=7)


def setup_joint_panels(fig, joint_pos, joint_names, layout, dt):
    """Create a 4x3 grid of axes on the right half of the figure and plot
    all 12 joint-angle time series once. Returns the axes grid and the list
    of vertical cursor lines (to be moved per frame).
    """
    T = joint_pos.shape[0]
    t_axis = np.arange(T) * dt

    gs = fig.add_gridspec(4, 6, wspace=0.45, hspace=0.55)
    ax_skel = fig.add_subplot(gs[:, 0:3], projection="3d")

    joint_axes = np.empty((4, 3), dtype=object)
    cursor_lines = []

    for li in range(4):
        rows_indices = [layout[li, ji] for ji in range(3) if layout[li, ji] >= 0]
        if not rows_indices:
            y_lo, y_hi = -1.0, 1.0
        else:
            vals = joint_pos[:, rows_indices]
            y_lo = float(np.min(vals)) - 0.05
            y_hi = float(np.max(vals)) + 0.05
            if y_hi - y_lo < 1e-3:
                y_lo, y_hi = y_lo - 0.1, y_hi + 0.1

        for ji in range(3):
            ax = fig.add_subplot(gs[li, 3 + ji])
            joint_axes[li, ji] = ax
            j_idx = layout[li, ji]
            if j_idx < 0:
                ax.set_visible(False)
                continue
            jname = joint_names[j_idx]
            ax.plot(t_axis, joint_pos[:, j_idx], linewidth=1.2, color="tab:blue")
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlim(t_axis[0], t_axis[-1] if T > 1 else 1.0)
            ax.set_title(jname, fontsize=8)
            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, alpha=0.3)
            if li == 3:
                ax.set_xlabel("t (s)", fontsize=8)
            if ji == 0:
                ax.set_ylabel("q (rad)", fontsize=8)

            cur = ax.axvline(t_axis[0], color="red", linewidth=1.0)
            cursor_lines.append((cur, t_axis))

    return ax_skel, joint_axes, cursor_lines


def update_cursors(frame_idx, cursor_lines):
    for cur, t_axis in cursor_lines:
        if frame_idx < len(t_axis):
            cur.set_xdata([t_axis[frame_idx], t_axis[frame_idx]])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Animate retarget.py output: body-frame skeleton + joint time series."
    )
    parser.add_argument("--npz", required=True, help="Path to retargeted .npz from retarget.py")
    parser.add_argument("--interval", type=int, default=33, help="Animation frame delay in ms")
    parser.add_argument("--elev", type=float, default=20.0,
                        help="Initial camera elevation (you can rotate with the mouse afterwards).")
    parser.add_argument("--azim", type=float, default=-100.0,
                        help="Initial camera azimuth (you can rotate with the mouse afterwards).")
    parser.add_argument("--show-labels", action="store_true", help="Label each foot")
    parser.add_argument("--show-legend", action="store_true", default=True,
                        help="Show legend on the skeleton panel (first frame).")
    parser.add_argument("--save", type=str, default=None,
                        help="Optional path to save as .mp4 / .gif instead of showing live.")
    args = parser.parse_args()

    try:
        data = load_retarget_npz(args.npz)
    except Exception as e:
        print(str(e), file=sys.stderr); sys.exit(1)

    root_pos = data["root_pos"]
    root_rot = data["root_rot"]
    joint_pos = data["joint_pos"]
    joint_names = data["joint_names"]
    foot_pos = data["foot_pos"]
    foot_pos_local = data["foot_pos_local"]
    foot_names = data["foot_names"]
    fps = data["fps"]
    dt = data["dt"]
    T = root_pos.shape[0]

    # Report what we loaded. No longer assumes the root is pinned.
    print(f"Loaded '{args.npz}'")
    print(f"  frames    : {T}  (dt={dt:.5f}s, fps={fps:.2f})")
    print(f"  root_pos  : {root_pos.shape}   "
          f"(range x={root_pos[:,0].ptp():.3g} "
          f"y={root_pos[:,1].ptp():.3g} z={root_pos[:,2].ptp():.3g})")
    # Quick summary of root orientation variability: geodesic spread of
    # root_rot around its mean, using |1 - |<q, q_mean>||.
    q_mean = _normalize_quat(root_rot.mean(axis=0))
    dots = np.abs(np.einsum("ti,i->t", _normalize_quat(root_rot), q_mean))
    rot_spread_rad = float(2.0 * np.arccos(np.clip(np.min(dots), -1.0, 1.0)))
    print(f"  root_rot  : {root_rot.shape}   "
          f"(max angle off mean ≈ {np.degrees(rot_spread_rad):.2f} deg)")
    print(f"  joint_pos : {joint_pos.shape}  names={joint_names}")
    print(f"  foot_pos  : {foot_pos.shape}   names={foot_names}")

    # Targets live in the body frame already; transform achieved feet into
    # the same (per-frame) body frame using root_pos and root_rot.
    foot_target_local = foot_pos_local
    foot_achieved_local = world_to_body_points(foot_pos, root_pos, root_rot)

    # Axis bounds come from body-frame quantities only — the root sits at
    # the origin of the body frame, so we just need the union of target and
    # achieved foot positions (plus the origin for padding).
    skeleton_bounds = compute_bounds(
        [np.zeros((1, 3), dtype=np.float32),
         foot_target_local, foot_achieved_local],
        pad_frac=0.15,
    )

    layout = build_joint_layout(joint_names)

    fig = plt.figure(figsize=(16, 7.5))
    ax_skel, joint_axes, cursor_lines = setup_joint_panels(
        fig, joint_pos, joint_names, layout, dt
    )
    ax_skel.view_init(elev=args.elev, azim=args.azim)
    try:
        ax_skel.mouse_init()
    except Exception:
        pass

    fig.suptitle(
        "Retargeted quadruped — body-frame skeleton (left, drag to rotate) "
        "and joint trajectories (right)",
        fontsize=12,
    )

    def _update(frame_idx):
        draw_skeleton_panel(
            ax_skel, frame_idx, foot_target_local,
            foot_achieved_local, foot_names, skeleton_bounds, args, fps,
            root_pos=root_pos, root_rot=root_rot,
        )
        update_cursors(frame_idx, cursor_lines)
        return []

    ani = FuncAnimation(
        fig, _update, frames=T,
        interval=args.interval, blit=False, repeat=True,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if args.save is not None:
        print(f"[saving] writing animation to {args.save} ...")
        if args.save.lower().endswith(".gif"):
            ani.save(args.save, writer="pillow", fps=max(1, int(round(fps))))
        else:
            ani.save(args.save, fps=max(1, int(round(fps))))
        print("[saving] done.")
    else:
        plt.show()


if __name__ == "__main__":
    main()