import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.ndimage import uniform_filter1d  # uniform kernel filter

parser = argparse.ArgumentParser()
parser.add_argument("--motion", type=str, required=True)
parser.add_argument("--cutoff_hz", type=float, default=3.0,
                    help="Low-pass filter cutoff in Hz. 2-4 Hz is typical for walking gait. "
                         "Lower = smoother but loses fast motion. Higher = keeps more detail.")
parser.add_argument("--savgol_win", type=int, default=11,
                    help="Savitzky-Golay window (odd number). Larger = smoother.")
parser.add_argument("--out", type=str, default=None,
                    help="Output .npz path. Default: <input>_smoothed.npz")
parser.add_argument("--upsample", type=int, default=1,
                    help="Upsample factor. 2 = double frame rate via cubic spline.")
args = parser.parse_args()

JOINT_NAMES = [
    "LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA",
    "LF_HFE", "LH_HFE", "RF_HFE", "RH_HFE",
    "LF_KFE", "LH_KFE", "RF_KFE", "RH_KFE",
]

data = np.load(args.motion, allow_pickle=True)
frame_dt = float(data["frame_duration"])
raw = np.asarray(data["joint_pos"], dtype=np.float64)  # (T, 12)
T, J = raw.shape    # time, joint
t_raw = np.arange(T) * frame_dt
fps = 1.0 / frame_dt

print(f"[Diagnose] {T} frames  dt={frame_dt:.4f}s  fps={fps:.1f}  duration={T * frame_dt:.2f}s")
print(f"[Diagnose] Joint data shape: {raw.shape}")

print("\n── Per-joint diagnosis ──")
print(f"  {'joint':>8}  {'mean':>8}  {'std':>7}  {'range':>7}  "
      f"{'step_mean':>10}  {'step_max':>9}  flag")

max_step_all = 0.0
for j in range(J):
    q = raw[:, j]
    steps = np.abs(np.diff(q))  # frame-to-frame jumps
    s_mean = steps.mean()
    s_max = steps.max()
    max_step_all = max(max_step_all, s_max)
    # Flag if max step > 10× mean step (suggests discrete keyframes / snapping)
    flag = "← STAIRCASE?" if s_max > 5 * s_mean and s_mean < 0.01 else ""
    print(f"  {JOINT_NAMES[j]:>8}  {q.mean():+8.4f}  {q.std():7.4f}  "
          f"{q.max() - q.min():7.4f}  {s_mean:10.5f}  {s_max:9.5f}  {flag}")

print(f"\n  Max single-frame jump across all joints: {max_step_all:.5f} rad")
print(f"  At fps={fps:.1f}, this corresponds to velocity: "
      f"{max_step_all / frame_dt:.2f} rad/s")
if max_step_all / frame_dt > 5.0:
    print("  [WARN] Instantaneous velocity > 5 rad/s detected — motion data is noisy/discontinuous")

# Autocorrelation check — smooth signals have high short-lag autocorrelation
print("\n── Smoothness check (lag-1 autocorrelation, should be > 0.99 for smooth motion) ──")
for j in range(J):
    q = raw[:, j]
    q_n = q - q.mean()
    ac = np.correlate(q_n, q_n, mode="full")
    ac /= ac[len(ac) // 2]  
    lag1 = ac[len(ac) // 2 + 1]
    flag = "← NOISY" if lag1 < 0.97 else ""
    print(f"  {JOINT_NAMES[j]:>8}  autocorr_lag1={lag1:.4f}  {flag}")

print(f"\n── Smoothing  cutoff={args.cutoff_hz}Hz  "
      f"savgol_win={args.savgol_win}  upsample={args.upsample}x ──")

# Step 1: Fit cubic splines through keyframes (removes staircase)
cs = CubicSpline(t_raw, raw, bc_type="periodic" if False else "not-a-knot")
# Upsample if requested
new_dt = frame_dt / args.upsample
t_new = np.arange(0, t_raw[-1] + new_dt * 0.5, new_dt)
splined = cs(t_new).astype(np.float32)  # (T_new, 12)

# Step 2: Savitzky-Golay filter (smooths while preserving peaks)
win = args.savgol_win
if win >= splined.shape[0]:
    win = splined.shape[0] // 4 * 2 - 1  # must be odd and < T
if win < 3:
    win = 3
sg_smooth = savgol_filter(splined, window_length=win, polyorder=3, axis=0)

# Step 3: Butterworth low-pass (removes high-freq noise)
nyq = 0.5 / new_dt
cutoff = min(args.cutoff_hz / nyq, 0.99)
b, a = butter(4, cutoff, btype="low")
smooth = filtfilt(b, a, sg_smooth, axis=0).astype(np.float32)

print(f"  Output: {smooth.shape[0]} frames  dt={new_dt:.4f}s  "
      f"duration={smooth.shape[0] * new_dt:.2f}s")

# Smoothness check on output
print("\n── Smoothness check on smoothed data ──")
for j in range(J):
    steps_s = np.abs(np.diff(smooth[:, j]))
    steps_r = np.abs(np.diff(raw[:, j]))
    reduction = (1 - steps_s.mean() / (steps_r.mean() + 1e-9)) * 100
    print(f"  {JOINT_NAMES[j]:>8}  "
          f"raw_step={steps_r.mean():.5f}  "
          f"smooth_step={steps_s.mean():.5f}  "
          f"reduction={reduction:.1f}%")

out_dir = args.motion.rsplit("/", 1)[0] if "/" in args.motion else "."

# Plot 1: Position comparison for all joints
fig, axes = plt.subplots(6, 2, figsize=(16, 20))
fig.suptitle("Motion Smoothing: Raw vs Smoothed", fontsize=14)
for j, ax in enumerate(axes.flat):
    if j >= J:
        ax.axis("off");
        continue
    t_plot = t_new[:600]  # first 12 seconds
    ax.plot(t_raw[:int(12 / frame_dt)], raw[:int(12 / frame_dt), j],
            "r-", alpha=0.5, linewidth=1, label="raw")
    ax.plot(t_plot, smooth[:600, j],
            "b-", linewidth=1.5, label="smoothed")
    ax.set_title(JOINT_NAMES[j])
    ax.set_xlabel("t (s)")
    ax.set_ylabel("rad")
    if j == 0:
        ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{out_dir}/motion_smoothing_comparison.png", dpi=120)
print(f"\n[Plot] Saved motion_smoothing_comparison.png")
plt.close()

# Plot 2: Velocity comparison (key diagnostic)
fig, axes = plt.subplots(4, 3, figsize=(18, 12))
fig.suptitle("Joint Velocities: Raw vs Smoothed\n"
             "(should look like smooth sine waves, not noise)", fontsize=13)
raw_vel = np.diff(raw, axis=0) / frame_dt  # (T-1, 12)
smooth_vel = np.diff(smooth, axis=0) / new_dt  # (T_new-1, 12)
for j, ax in enumerate(axes.flat):
    n_show = min(300, len(raw_vel))
    ax.plot(t_raw[1:n_show + 1], raw_vel[:n_show, j],
            "r-", alpha=0.4, linewidth=0.8, label="raw vel")
    ax.plot(t_new[1:n_show * args.upsample + 1],
            smooth_vel[:n_show * args.upsample, j],
            "b-", linewidth=1.0, label="smooth vel")
    ax.set_title(JOINT_NAMES[j])
    ax.axhline(0, color="k", linewidth=0.5)
    if j == 0:
        ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{out_dir}/velocity_comparison.png", dpi=120)
print(f"[Plot] Saved velocity_comparison.png")
plt.close()

# Plot 3: Power spectral density — shows dominant gait frequency
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Power Spectral Density (what frequencies are in the motion?)\n"
             "Expected: strong peak at gait frequency (~1-3 Hz), "
             "little above cutoff", fontsize=11)
for j, ax in enumerate(axes.flat):
    if j >= 6:
        ax.axis("off");
        continue
    q = raw[:, j] - raw[:, j].mean()
    fft = np.abs(np.fft.rfft(q))
    freqs = np.fft.rfftfreq(len(q), d=frame_dt)
    ax.semilogy(freqs, fft + 1e-10, "r-", alpha=0.7, linewidth=0.8, label="raw")
    q_s = smooth[:, j] - smooth[:, j].mean()
    fft_s = np.abs(np.fft.rfft(q_s))
    freqs_s = np.fft.rfftfreq(len(q_s), d=new_dt)
    ax.semilogy(freqs_s, fft_s + 1e-10, "b-", alpha=0.7, linewidth=0.8, label="smooth")
    ax.axvline(args.cutoff_hz, color="g", linestyle="--", linewidth=1, label=f"cutoff={args.cutoff_hz}Hz")
    ax.set_xlim(0, min(fps / 2, 15))
    ax.set_title(JOINT_NAMES[j])
    ax.set_xlabel("Hz")
    if j == 0:
        ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{out_dir}/psd_analysis.png", dpi=120)
print(f"[Plot] Saved psd_analysis.png")
plt.close()

out_path = args.out or args.motion.replace(".npz", "_smoothed.npz")

# Preserve all original keys, replace joint_pos and frame_duration
save_dict = {}
for k in data.files:
    if k == "joint_pos":
        save_dict["joint_pos"] = smooth
    elif k == "frame_duration":
        save_dict["frame_duration"] = np.float32(new_dt)
    else:
        # Upsample other arrays (root_pos, root_rpy) if they exist
        if args.upsample > 1 and data[k].ndim >= 1 and len(data[k]) == T:
            try:
                arr = np.asarray(data[k], dtype=np.float64)
                cs_arr = CubicSpline(t_raw, arr)
                arr_up = cs_arr(t_new).astype(np.float32)
                bw, aw = butter(4, cutoff, btype="low")
                arr_up = filtfilt(bw, aw, arr_up, axis=0).astype(np.float32)
                save_dict[k] = arr_up
                print(f"  Also smoothed/upsampled: {k}  {arr.shape} → {arr_up.shape}")
            except Exception:
                save_dict[k] = data[k]
        else:
            save_dict[k] = data[k]

save_dict["frame_duration"] = np.float32(new_dt)
np.savez(out_path, **save_dict)
print(f"\n[Save] Smoothed motion → {out_path}")
print(f"  Frames: {T} → {smooth.shape[0]}  "
      f"dt: {frame_dt:.4f} → {new_dt:.4f}  "
      f"duration: {T * frame_dt:.2f}s")

print("\n══ SUMMARY ══")
print(f"  Raw motion:  {T} frames @ {fps:.1f}fps  "
      f"max_step={max_step_all:.4f} rad  "
      f"max_vel={max_step_all / frame_dt:.2f} rad/s")
smooth_steps = np.abs(np.diff(smooth, axis=0)).max()
print(f"  Smoothed:    {smooth.shape[0]} frames @ {1 / new_dt:.1f}fps  "
      f"max_step={smooth_steps:.4f} rad  "
      f"max_vel={smooth_steps / new_dt:.2f} rad/s")
print(f"\n  Next step: use {out_path} for BC/RL training")
print(f"  Command: python train_motion_imitation.py --motion {out_path} --mode bc")
