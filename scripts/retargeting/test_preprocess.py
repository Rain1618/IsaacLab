"""
test_preprocess.py — Validate CSV->NPZ pipeline with synthetic quadruped motion.
"""

import numpy as np
import tempfile
import os
import sys

BODY_PARTS = [
    "neck_base", "back_end", "back_middle",
    "front_left_thai", "front_left_knee", "front_left_paw",
    "front_right_thai", "front_right_knee", "front_right_paw",
    "back_left_paw", "back_left_thai",
    "back_right_thai", "back_left_knee", "back_right_knee", "back_right_paw",
]

def generate_synthetic_csv(path: str, T: int = 100, fps: float = 30.0):
    dt = 1.0 / fps
    t = np.arange(T) * dt

    cx = 0.5 * t
    cy = np.zeros(T)
    cz = np.full(T, 0.45) + 0.02 * np.sin(2 * np.pi * 2.0 * t)

    yaw = 0.05 * np.sin(2 * np.pi * 0.5 * t)
    fwd_x = np.cos(yaw)
    fwd_y = np.sin(yaw)

    kp_data = {}
    body_len = 0.4
    kp_data["neck_base"]   = np.stack([cx + body_len * fwd_x, cy + body_len * fwd_y, cz], axis=-1)
    kp_data["back_end"]    = np.stack([cx - body_len * fwd_x, cy - body_len * fwd_y, cz], axis=-1)
    kp_data["back_middle"] = np.stack([cx, cy, cz], axis=-1)

    hw = 0.15
    leg_len = 0.4
    phases = [0.0, np.pi, np.pi, 0.0]

    leg_configs = [
        ("front_left",  +body_len * 0.7, +hw),
        ("front_right", +body_len * 0.7, -hw),
        ("back_left",   -body_len * 0.7, +hw),
        ("back_right",  -body_len * 0.7, -hw),
    ]

    for i, (prefix, dx, dy) in enumerate(leg_configs):
        hip_x = cx + dx * fwd_x - dy * fwd_y
        hip_y = cy + dx * fwd_y + dy * fwd_x
        hip_z = cz
        knee_z = cz - leg_len * 0.5
        paw_z = np.maximum(0.0, 0.05 * np.sin(2 * np.pi * 2.0 * t + phases[i]))

        kp_data[f"{prefix}_thai"] = np.stack([hip_x, hip_y, hip_z], axis=-1)
        kp_data[f"{prefix}_knee"] = np.stack([hip_x, hip_y, knee_z], axis=-1)
        kp_data[f"{prefix}_paw"]  = np.stack([hip_x, hip_y, paw_z], axis=-1)

    header0_parts = []
    header1_parts = []
    for bp in BODY_PARTS:
        for _ in range(4):
            header0_parts.append(bp)
        header1_parts.extend(["x", "y", "likelihood", "z"])

    lines = []
    lines.append("(" + ",".join(header0_parts) + ")")
    lines.append("(" + ",".join(header1_parts) + ")")

    for frame in range(T):
        vals = []
        for bp in BODY_PARTS:
            pos = kp_data[bp][frame]
            vals.extend([f"{pos[0]:.6f}", f"{pos[1]:.6f}", "0.990000", f"{pos[2]:.6f}"])
        lines.append(",".join(vals))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[Test] Generated synthetic CSV: {T} frames -> {path}")
    return T


def test_pipeline():
    from scripts.retargeting.preprocess_csv import csv_to_npz

    fps = 30.0
    T_expected = 100

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_motion.csv")
        npz_path = os.path.join(tmpdir, "test_motion.npz")

        generate_synthetic_csv(csv_path, T=T_expected, fps=fps)
        csv_to_npz(csv_path, npz_path, fps=fps, lh_threshold=0.5)

        data = dict(np.load(npz_path))

        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)

        checks = [
            ("root_pos",     (T_expected, 3)),
            ("root_rot",     (T_expected, 4)),
            ("foot_pos",     (T_expected, 4, 3)),
            ("root_lin_vel", (T_expected, 3)),
            ("root_ang_vel", (T_expected, 3)),
        ]
        all_pass = True
        for key, expected_shape in checks:
            actual = data[key].shape
            ok = actual == expected_shape
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {key:15s} : expected {expected_shape}, got {actual}")
            all_pass = all_pass and ok

        qnorms = np.linalg.norm(data["root_rot"], axis=-1)
        qnorm_ok = np.allclose(qnorms, 1.0, atol=1e-3)
        print(f"  [{'PASS' if qnorm_ok else 'FAIL'}] quat norms   : "
              f"mean={qnorms.mean():.6f}, std={qnorms.std():.6f}")
        all_pass = all_pass and qnorm_ok

        rz = data["root_pos"][:, 2]
        height_ok = rz.min() > 0.0 and rz.max() < 2.0
        print(f"  [{'PASS' if height_ok else 'FAIL'}] root height  : "
              f"[{rz.min():.3f}, {rz.max():.3f}] m")
        all_pass = all_pass and height_ok

        fz = data["foot_pos"][:, :, 2]
        foot_ok = fz.min() >= -0.01
        print(f"  [{'PASS' if foot_ok else 'FAIL'}] foot z-range  : "
              f"[{fz.min():.3f}, {fz.max():.3f}] m")
        all_pass = all_pass and foot_ok

        # Verify rotation matrices are valid (orthogonal) by round-tripping
        from scripts.retargeting.preprocess_csv import _rotmat_to_quat_batch, _quat_mul_batch_np
        q = data["root_rot"]
        # q * conj(q) should be [1, 0, 0, 0]
        q_conj = q.copy()
        q_conj[:, 1:] *= -1
        identity = _quat_mul_batch_np(q, q_conj)
        id_ok = np.allclose(identity[:, 0], 1.0, atol=1e-3) and np.allclose(identity[:, 1:], 0.0, atol=1e-3)
        print(f"  [{'PASS' if id_ok else 'FAIL'}] q*conj(q)=id : "
              f"w_mean={identity[:, 0].mean():.6f}")
        all_pass = all_pass and id_ok

        print(f"  [INFO] fps = {data['fps'][0]:.1f}, dt = {data['dt'][0]:.6f}")

        print("=" * 60)
        if all_pass:
            print("ALL CHECKS PASSED")
        else:
            print("SOME CHECKS FAILED")
            sys.exit(1)
        print("=" * 60)


if __name__ == "__main__":
    test_pipeline()