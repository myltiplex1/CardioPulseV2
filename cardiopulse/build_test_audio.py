import os
import numpy as np
import wave

from spectrogram_utils import TARGET_SR, load_and_filter

# --- CONFIGURATION ---
SOURCE_DIR = "Organized_Heart_Audio/Normal"
OUTPUT_FILENAME = "test_audio_5min_normal.wav"
TARGET_DURATION_SEC = 300  # 5 minutes


def save_wav(filepath, float_audio, sr=TARGET_SR):
    int16_audio = np.clip(float_audio * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_audio.tobytes())


def main():
    if not os.path.isdir(SOURCE_DIR):
        print(f"ERROR: '{SOURCE_DIR}' not found.")
        return

    files = sorted(f for f in os.listdir(SOURCE_DIR) if f.endswith(".wav"))
    if not files:
        print(f"ERROR: no .wav files found in '{SOURCE_DIR}'.")
        return

    print(f"Building a ~{TARGET_DURATION_SEC}s test file from '{SOURCE_DIR}'...")

    segments = []
    total_duration = 0.0
    files_used = 0

    for f in files:
        if total_duration >= TARGET_DURATION_SEC:
            break
        filepath = os.path.join(SOURCE_DIR, f)
        y = load_and_filter(filepath, TARGET_SR)  # same preprocessing used everywhere else
        segments.append(y)
        total_duration += len(y) / TARGET_SR
        files_used += 1

    combined = np.concatenate(segments)

    # Trim to exactly the target duration
    target_samples = int(TARGET_DURATION_SEC * TARGET_SR)
    if len(combined) > target_samples:
        combined = combined[:target_samples]

    save_wav(OUTPUT_FILENAME, combined)

    actual_duration = len(combined) / TARGET_SR
    print(f"Used {files_used} source files.")
    print(f"Saved '{OUTPUT_FILENAME}' -- {actual_duration:.1f} seconds.")


if __name__ == "__main__":
    main()
