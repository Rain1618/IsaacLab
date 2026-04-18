"""
preprocess_csv_fixed.py — Convert animal motion-capture CSV to retargeting-ready .npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


FOOT_NAMES = [
    "front_left_paw",
    "front_right_paw",
    "back_left_paw",
    "back_right_paw",
]

TRUNK_NAMES = ["neck_base", "back_middle", "back_end"]


def parse_mocap_csv(csv_path: str) -> dict[str, np.ndarray]:
    with open(csv_path, "r") as f:
        lines = f.readlines()

    header_parts = lines[0].strip().split(",")
    header_fields = lines[1].strip().split(",")

    col_map: dict[str, dict[str, int]] = {}
    for col_i, (part, field) in enumerate(zip(header_parts, header_fields)):
        part = part.strip()
        field = field.strip()
        col_map.setdefault(part, {})[field] = col_i

    data_lines = lines[2:]
    raw = np.zeros((len(data_lines), len(header_parts)), dtype=np.float64)
    
    for t, line in enumerate(data_lines):
        raw[t] = [float(v.strip()) for v in line.strip().split(",")]

    keypoints: dict[str, np.ndarray] = {}
    for part, fields in col_map.items():
        keypoints[part] = np.stack(
            [
                raw[:, fields["x"]],
                raw[:, fields["y"]],
                raw[:, fields["likelihood"]],
                raw[:, fields["z"]],
            ],
            axis=-1,
        ).astype(np.float32)

    print(f"[CSV] Parsed {raw.shape[0]} frames, {len(keypoints)} keypoints")
    return keypoints


def xyz(kp: np.ndarray) -> np.ndarray:
    return kp[:, [0, 1, 3]]


def smooth_series(x: np.ndarray, window: int = 7) -> np.ndarray:
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


def smooth_low_confidence(keypoints: dict[str, np.ndarray], threshold: float = 0.5) -> dict[str, np.ndarray]:
    for name, kp in keypoints.items():
        lh = kp[:, 2]
        low = lh < threshold
        if not np.any(low):
            continue
        good_idx = np.where(~low)[0]
        bad_idx = np.where(low)[0]
        if len(good_idx) < 2:
            continue
        print(f"  [Smooth] {name}: interpolating {len(bad_idx)}/{len(kp)} low-confidence frames")
        for col in [0, 1, 3]:
            kp[bad_idx, col] = np.interp(bad_idx, good_idx, kp[good_idx, col])
    return keypoints


def temporal_smooth_keypoints(keypoints: dict[str, np.ndarray], window: int = 7) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, kp in keypoints.items():
        kp_sm = kp.copy()
        kp_sm[:, [0, 1, 3]] = smooth_series(kp[:, [0, 1, 3]], window=window)
        out[name] = kp_sm.astype(np.float32)
    return out


def estimate_root_pos(keypoints: dict[str, np.ndarray]) -> np.ndarray:
    trunk_xyz = np.stack([xyz(keypoints[n]) for n in TRUNK_NAMES], axis=1)
    return trunk_xyz.mean(axis=1).astype(np.float32)


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8, None)


def _rotmat_to_quat_batch(R: np.ndarray) -> np.ndarray:
    B = R.shape[0]
    q = np.zeros((B, 4), dtype=np.float64)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    d0, d1, d2 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]

    m1 = tr > 0
    s1 = np.sqrt(tr[m1] + 1.0) * 2.0
    q[m1, 0] = 0.25 * s1
    q[m1, 1] = (R[m1, 2, 1] - R[m1, 1, 2]) / s1
    q[m1, 2] = (R[m1, 0, 2] - R[m1, 2, 0]) / s1
    q[m1, 3] = (R[m1, 1, 0] - R[m1, 0, 1]) / s1

    m2 = (~m1) & (d0 >= d1) & (d0 >= d2)
    s2 = np.sqrt(1.0 + d0[m2] - d1[m2] - d2[m2]) * 2.0
    q[m2, 0] = (R[m2, 2, 1] - R[m2, 1, 2]) / s2
    q[m2, 1] = 0.25 * s2
    q[m2, 2] = (R[m2, 0, 1] + R[m2, 1, 0]) / s2
    q[m2, 3] = (R[m2, 0, 2] + R[m2, 2, 0]) / s2

    m3 = (~m1) & (~m2) & (d1 >= d2)
    s3 = np.sqrt(1.0 + d1[m3] - d0[m3] - d2[m3]) * 2.0
    q[m3, 0] = (R[m3, 0, 2] - R[m3, 2, 0]) / s3
    q[m3, 1] = (R[m3, 0, 1] + R[m3, 1, 0]) / s3
    q[m3, 2] = 0.25 * s3
    q[m3, 3] = (R[m3, 1, 2] + R[m3, 2, 1]) / s3

    m4 = (~m1) & (~m2) & (~m3)
    s4 = np.sqrt(1.0 + d2[m4] - d0[m4] - d1[m4]) * 2.0
    q[m4, 0] = (R[m4, 1, 0] - R[m4, 0, 1]) / s4
    q[m4, 1] = (R[m4, 0, 2] + R[m4, 2, 0]) / s4
    q[m4, 2] = (R[m4, 1, 2] + R[m4, 2, 1]) / s4
    q[m4, 3] = 0.25 * s4

    q /= np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    return q.astype(np.float32)


def ensure_quaternion_continuity(quats: np.ndarray) -> np.ndarray:
    q = quats.copy().astype(np.float32)
    q /= np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)
    for t in range(1, q.shape[0]):
        if np.dot(q[t - 1], q[t]) < 0.0:
            q[t] *= -1.0
    return q


def estimate_root_rot(keypoints: dict[str, np.ndarray]) -> np.ndarray:
    neck = xyz(keypoints["neck_base"])
    back = xyz(keypoints["back_end"])
    lhip = xyz(keypoints["front_left_thai"])
    rhip = xyz(keypoints["front_right_thai"])

    fwd = _safe_normalize(neck - back)
    lat = _safe_normalize(rhip - lhip)
    z_axis = _safe_normalize(np.cross(fwd, lat))
    y_axis = _safe_normalize(np.cross(z_axis, fwd))

    R = np.stack([fwd, y_axis, z_axis], axis=-1)
    q = _rotmat_to_quat_batch(R)
    return ensure_quaternion_continuity(q)


def extract_foot_positions(keypoints: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([xyz(keypoints[n]) for n in FOOT_NAMES], axis=1).astype(np.float32)


def csv_to_npz(csv_path: str, output_path: str, fps: float, lh_threshold: float = 0.5) -> None:
    dt = 1.0 / fps
    keypoints = parse_mocap_csv(csv_path)
    keypoints = smooth_low_confidence(keypoints, threshold=lh_threshold)
    keypoints = temporal_smooth_keypoints(keypoints, window=7)

    root_pos = smooth_series(estimate_root_pos(keypoints), window=5)
    root_rot = estimate_root_rot(keypoints)
    foot_pos = smooth_series(extract_foot_positions(keypoints), window=5)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        root_pos=root_pos.astype(np.float32),
        root_rot=root_rot.astype(np.float32),
        foot_pos=foot_pos.astype(np.float32),
        fps=np.array([fps], dtype=np.float32),
        dt=np.array([dt], dtype=np.float32),
    )
    print(f"[Preprocess] Saved {root_pos.shape[0]} frames @ {fps} Hz -> '{out_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert animal motion-capture CSV to retargeting .npz")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--lh-threshold", type=float, default=0.5)
    args = parser.parse_args()
    csv_to_npz(args.csv, args.output, args.fps, args.lh_threshold)
