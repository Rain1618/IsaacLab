import json
import numpy as np
import os
import argparse

def zero_out_column(frames, column_index):
    """
    Replaces all values in a specified column of the frames array with zeroes.
    """
    if 0 <= column_index < frames.shape[1]:
        frames[:, column_index+1:] = 0.0
        print(f"[INFO] Zeroed out column index: {column_index}")
    else:
        print(f"[WARNING] Column index {column_index} is out of bounds (0 to {frames.shape[1]-1}). Skipping.")
    return frames

def load_anymal_motion(json_path, save_path, zero_col_idx=None):
    print(f"[INFO] Loading JSON from: {json_path}")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Input JSON not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    frames = np.array(data["Frames"], dtype=np.float32)
    frame_duration = float(data["FrameDuration"])

    print(f"[INFO] Loaded {frames.shape[0]} frames with {frames.shape[1]} values each.")

    # Apply the zero-out function if a column index was passed
    if zero_col_idx is not None:
        frames = zero_out_column(frames, zero_col_idx)

    root_pos = frames[:, 0:3]
    root_rpy = frames[:, 3:7]
    joint_pos = frames[:, 7:]

    print(f"[INFO] root_pos shape:  {root_pos.shape}")
    print(f"[INFO] root_rpy shape:  {root_rpy.shape}")
    print(f"[INFO] joint_pos shape: {joint_pos.shape}")

    out_dir = os.path.dirname(os.path.abspath(save_path))
    if out_dir:  # Ensure out_dir isn't empty if save_path is just a filename
        os.makedirs(out_dir, exist_ok=True)

    np.savez(save_path,
             frame_duration=frame_duration,
             root_pos=root_pos,
             root_rpy=root_rpy,
             joint_pos=joint_pos)
    print(f"[CHECK] Actual output path: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ANYmal motion JSON to NPZ format.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="../pace.json", 
        help="Path to the input JSON file."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="anymal_motion_for_isaaclab.npz", 
        help="Path to save the output NPZ file."
    )
    parser.add_argument(
        "--zero-col", 
        type=int, 
        default=None, 
        help="The index of the column in 'frames' to replace with zeroes."
    )

    args = parser.parse_args()

    load_anymal_motion(
        json_path=args.input,
        save_path=args.output,
        zero_col_idx=args.zero_col
    )