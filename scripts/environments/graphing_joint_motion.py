import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Dummy constants to match the environment structure
ISAAC_DEFAULT = np.zeros(12, dtype=np.float32)
SIGN_FLIP = np.ones(12, dtype=np.float32)

def create_dummy_clip():
    T = 20
    frame_dt = 0.05
    duration = T * frame_dt
    
    # Create periodic dummy data for joint 0 (e.g., a walking gait phase)
    t_raw = np.linspace(0, duration, T, endpoint=False)
    j0 = np.sin(2 * np.pi * t_raw / duration) + 0.3 * np.cos(4 * np.pi * t_raw / duration)
    
    # Fill all 12 joints
    raw_isaac = np.zeros((T, 12))
    raw_isaac[:, 0] = j0
    
    clip_mean = raw_isaac.mean(axis=0)
    delta = (raw_isaac - clip_mean) * SIGN_FLIP[None, :]
    joint_pos = delta + ISAAC_DEFAULT[None, :]
    
    # 1. Append the first frame to the very end to create a closed loop
    joint_pos_looped = np.vstack((joint_pos, joint_pos[0]))
    
    # 2. Create the time array for the keyframes (length T + 1)
    t_frames = np.linspace(0, duration, T + 1)
    
    # 3. Fit the periodic cubic spline across all joints (axis=0)
    cs = CubicSpline(t_frames, joint_pos_looped, axis=0, bc_type='periodic')
    
    return {
        "joint_pos": joint_pos, 
        "frame_dt": frame_dt, 
        "T": T, 
        "duration": duration,
        "cs": cs,
        "t_frames": t_frames,
        "joint_pos_looped": joint_pos_looped
    }

clip = create_dummy_clip()

# Simulate physics engine queries over 3 full motion cycles
eval_dt = 0.01  # Higher resolution than frame_dt to show smoothness
t_eval = np.arange(0, clip["duration"] * 3, eval_dt)

def get_joint_target_batch(clip: dict, t_arr: np.ndarray) -> np.ndarray:
    t_wrapped = t_arr % clip["duration"]
    return clip["cs"](t_wrapped).astype(np.float32)

def get_ref_joint_velocities_batch(clip: dict, t_arr: np.ndarray) -> np.ndarray:
    t_wrapped = t_arr % clip["duration"]
    return clip["cs"](t_wrapped, 1).astype(np.float32)

pos_eval = get_joint_target_batch(clip, t_eval)
vel_eval = get_ref_joint_velocities_batch(clip, t_eval)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Position
ax1.plot(t_eval, pos_eval[:, 0], label="Spline Position (High Res)", color='blue')

# Overlay the original keyframes for reference
for i in range(3):
    ax1.plot(clip["t_frames"] + i * clip["duration"], clip["joint_pos_looped"][:, 0], 
             'ro', label="Mocap Keyframes" if i == 0 else "")

# Mark cycle wrapping boundaries
for i in range(4):
    ax1.axvline(i * clip["duration"], color='gray', linestyle='--', alpha=0.5)

ax1.set_ylabel("Position (rad)")
ax1.set_title("Joint 0: Target Position over 3 Looping Cycles (Spline vs Keyframes)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot Velocity
ax2.plot(t_eval, vel_eval[:, 0], label="Spline Velocity (Derivative)", color='green')
for i in range(4):
    ax2.axvline(i * clip["duration"], color='gray', linestyle='--', alpha=0.5)
    
ax2.set_ylabel("Velocity (rad/s)")
ax2.set_xlabel("Time (s)")
ax2.set_title("Joint 0: Target Velocity (True Derivative)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('joint_motion_spline.png', dpi=300)
plt.close()
print("Saved joint_motion_spline.png")