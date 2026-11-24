import os
import numpy as np
import librosa

from spectrogram_utils import TARGET_SR, DESPIKE_THRESHOLD

NOISE_DIR = "noise_clips"

# Same detection logic as despike_glitches, but for reporting rather than fixing
GLITCH_MAX_RUN = 3
GLITCH_NEIGHBOR_QUIET = 0.3


def find_glitches(y):
    abs_y = np.abs(y)
    candidates = np.where(abs_y >= DESPIKE_THRESHOLD)[0]
    if len(candidates) == 0:
        return []

    groups = [[candidates[0]]]
    for idx in candidates[1:]:
        if idx - groups[-1][-1] <= 2:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    glitches = []
    for group in groups:
        if len(group) > GLITCH_MAX_RUN:
            continue  # sustained loud region -- real audio, not a glitch
        start, end = group[0], group[-1]
        left, right = start - 1, end + 1
        if left < 0 or right >= len(y):
            continue
        if abs_y[left] < GLITCH_NEIGHBOR_QUIET and abs_y[right] < GLITCH_NEIGHBOR_QUIET:
            glitches.append(start / TARGET_SR)  # store timestamp in seconds

    return glitches


def check_file(filepath):
    y, sr = librosa.load(filepath, sr=TARGET_SR)
    duration = len(y) / sr

    clip_count = np.sum(np.abs(y) >= 0.999)
    clip_pct = 100 * clip_count / len(y)

    dc_offset = np.mean(y)
    rms = np.sqrt(np.mean(y ** 2))
    rms_db = 20 * np.log10(rms) if rms > 0 else float('-inf')
    peak_db = 20 * np.log10(np.max(np.abs(y))) if np.max(np.abs(y)) > 0 else float('-inf')

    glitches = find_glitches(y)

    return {
        "duration": duration,
        "clip_pct": clip_pct,
        "dc_offset": dc_offset,
        "rms_db": rms_db,
        "peak_db": peak_db,
        "n_glitches": len(glitches),
        "glitch_times": glitches,
    }


def main():
    if not os.path.exists(NOISE_DIR):
        print(f"ERROR: '{NOISE_DIR}' not found.")
        return

    files = sorted(f for f in os.listdir(NOISE_DIR) if f.endswith(".wav"))
    if not files:
        print(f"No .wav files found in '{NOISE_DIR}'.")
        return

    print("=" * 90)
    print(f" NOISE CLIP BATCH CHECK -- {len(files)} files")
    print("=" * 90)
    print(f"{'File':<28}{'Dur(s)':>8}{'Clip%':>9}{'DC':>10}{'RMS(dB)':>10}"
          f"{'Peak(dB)':>10}{'Glitches':>10}")
    print("-" * 90)

    flagged = []

    for f in files:
        stats = check_file(os.path.join(NOISE_DIR, f))
        flag = ""
        if stats["clip_pct"] > 1.0:
            flag += " CLIPPING"
        if abs(stats["dc_offset"]) > 0.05:
            flag += " DC-OFFSET"
        if stats["n_glitches"] > 5:
            flag += " MANY-GLITCHES"
        if flag:
            flagged.append((f, flag.strip()))

        print(f"{f:<28}{stats['duration']:>8.1f}{stats['clip_pct']:>9.3f}"
              f"{stats['dc_offset']:>10.4f}{stats['rms_db']:>10.1f}"
              f"{stats['peak_db']:>10.1f}{stats['n_glitches']:>10}")

    print("-" * 90)
    total_duration = sum(check_file(os.path.join(NOISE_DIR, f))["duration"] for f in files)
    print(f"Total noise pool duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")

    if flagged:
        print("\nFILES THAT MAY NEED A CLOSER LOOK:")
        for f, flag in flagged:
            print(f"  {f}: {flag}")
    else:
        print("\nNo files flagged -- all clips look within normal, healthy ranges.")

    print("=" * 90)


if __name__ == "__main__":
    main()
