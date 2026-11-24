import os
import json
import numpy as np

from spectrogram_utils import (wav_to_logmel, TARGET_SR, CHUNK_DURATION,
                               N_FFT, HOP_LENGTH, N_MELS, FMAX, N_FRAMES)

# --- CONFIGURATION ---
VAL_DIR = "split_dataset/val"
TEST_DIR = "split_dataset/test"
OUTPUT_DIR = "spectrogram_data"
CLASSES = ["Normal", "Murmur", "Extra_Systole", "Artifact"]

# NOTE: train is intentionally NOT precomputed here. It's built live, batch
# by batch, by train_data_generator.py -- that's what makes on-the-fly
# augmentation possible. val/test are never augmented, so precomputing them
# once (as done below) is fine and saves time during training/evaluation.


def build_split(split_dir, split_name):
    print(f"\nProcessing '{split_name}' split from '{split_dir}'...")
    X, y = [], []

    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  WARNING: '{class_dir}' not found, skipping.")
            continue

        files = sorted(f for f in os.listdir(class_dir) if f.endswith(".wav"))
        print(f"  {class_name}: {len(files)} files")

        for f in files:
            filepath = os.path.join(class_dir, f)
            try:
                spec = wav_to_logmel(filepath)
                X.append(spec)
                y.append(class_idx)
            except Exception as e:
                print(f"    ERROR processing {f}: {e}")

    X = np.array(X, dtype=np.float32)[..., np.newaxis]
    y = np.array(y, dtype=np.int64)
    print(f"  '{split_name}' shape: X={X.shape}, y={y.shape}")
    return X, y


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_val, y_val = build_split(VAL_DIR, "val")
    X_test, y_test = build_split(TEST_DIR, "test")

    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    label_map = {i: name for i, name in enumerate(CLASSES)}
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump(label_map, f, indent=2)

    meta = {
        "target_sr": TARGET_SR, "chunk_duration": CHUNK_DURATION,
        "n_fft": N_FFT, "hop_length": HOP_LENGTH, "n_mels": N_MELS,
        "fmax": FMAX, "n_frames": N_FRAMES,
        "input_shape": [N_MELS, N_FRAMES, 1],
        "note": "train split is NOT included here -- generated live by "
                "train_data_generator.py with on-the-fly augmentation.",
    }
    with open(os.path.join(OUTPUT_DIR, "spectrogram_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved val/test arrays + label_map.json + spectrogram_meta.json "
          f"to '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
