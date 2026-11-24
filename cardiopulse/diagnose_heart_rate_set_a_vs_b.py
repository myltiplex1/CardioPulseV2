import os
import re
import numpy as np
import librosa

from spectrogram_utils import TARGET_SR, apply_bandpass
from bpm_extraction import extract_envelope

RAW_DATA_DIR = "Organized_Heart_Audio"
MAX_FILES_PER_GROUP = 15
WINDOW_SEC = 3

SET_A_PATTERN = re.compile(r'^\w+__\d+\.wav$')


def classify_source(filename):
    return "Set A" if SET_A_PATTERN.match(filename) else "Set B"


def analyze_file(filepath):
    y, sr = librosa.load(filepath, sr=TARGET_SR)
    duration = len(y) / sr

    y_filtered = apply_bandpass(y, sr)
    rms = np.sqrt(np.mean(y_filtered ** 2))

    envelope = extract_envelope(y, sr)
    # "Peakiness" proxy: how much the envelope varies relative to its own
    # average level -- a low value suggests a flat, indistinct envelope
    # where S1/S2 don't stand out clearly from the background.
    peakiness = np.std(envelope) / (np.mean(envelope) + 1e-10)

    n_windows = max(0, int((duration - WINDOW_SEC) / 1.5) + 1) if duration >= WINDOW_SEC else 0

    return {
        "duration": duration,
        "rms": rms,
        "peakiness": peakiness,
        "n_windows": n_windows,
    }


def test_group(class_name, source_filter):
    class_dir = os.path.join(RAW_DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        return []

    all_files = sorted(f for f in os.listdir(class_dir) if f.endswith(".wav"))
    matching = [f for f in all_files if classify_source(f) == source_filter][:MAX_FILES_PER_GROUP]

    results = []
    for f in matching:
        try:
            stats = analyze_file(os.path.join(class_dir, f))
            results.append(stats)
        except Exception as e:
            print(f"    ERROR processing {f}: {e}")
    return results


def summarize(label, results):
    if not results:
        print(f"{label:<20}no files found")
        return

    durations = [r["duration"] for r in results]
    rmses = [r["rms"] for r in results]
    peakiness = [r["peakiness"] for r in results]
    n_windows = [r["n_windows"] for r in results]

    print(f"{label:<20}n={len(results):<4}"
          f"duration={np.mean(durations):>6.1f}s (std {np.std(durations):>5.1f})  "
          f"RMS={np.mean(rmses):>7.4f}  "
          f"peakiness={np.mean(peakiness):>6.2f}  "
          f"windows={np.mean(n_windows):>5.1f}")


def main():
    print("Diagnosing WHY Set B is harder than Set A for BPM extraction.\n")
    print(f"{'Group':<20}{'':<4}{'Avg duration':<20}{'Avg RMS':<9}{'Avg peakiness':<15}{'Avg #windows'}")
    print("-" * 95)

    summarize("Normal (Set A)", test_group("Normal", "Set A"))
    summarize("Normal (Set B)", test_group("Normal", "Set B"))
    summarize("Murmur (Set A)", test_group("Murmur", "Set A"))
    summarize("Murmur (Set B)", test_group("Murmur", "Set B"))

    print("-" * 95)
    print("\nWhat to look for:")
    print("  - Much shorter avg duration in Set B -> fewer windows -> less")
    print("    robust median -> explains lower confidence directly.")
    print("  - Lower RMS in Set B -> quieter recordings, possibly closer to")
    print("    the noise floor -> harder envelope extraction.")
    print("  - Lower peakiness in Set B -> S1/S2 less distinct from")
    print("    background in the envelope itself -> harder to time reliably.")


if __name__ == "__main__":
    main()
