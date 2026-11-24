import os
import csv
import numpy as np

from cardiac_parameters import extract_cardiac_parameters

TIMING_CSV_PATH = "set_a_timing.csv"
NORMAL_AUDIO_DIR = "Organized_Heart_Audio/Normal"


def load_ground_truth_sd_ratio(rows):
    """Computes the TRUE systolic/diastolic ratio per file directly from
    the expert S1/S2 labels -- S1-to-S2 gaps are true systole, S2-to-next-S1
    gaps are true diastole, no assumption needed since these are labeled."""
    header = rows[0]
    data = rows[1:]
    fname_idx = header.index("fname")
    sound_idx = header.index("sound")
    location_idx = header.index("location")

    file_events = {}
    for row in data:
        if len(row) <= max(fname_idx, sound_idx, location_idx):
            continue
        fname = row[fname_idx]
        sound = row[sound_idx]
        try:
            location = int(row[location_idx])
        except (ValueError, IndexError):
            continue
        file_events.setdefault(fname, []).append((location, sound))

    ground_truth = {}
    for fname, events in file_events.items():
        events.sort(key=lambda e: e[0])
        systolic_gaps, diastolic_gaps = [], []
        for i in range(len(events) - 1):
            loc1, sound1 = events[i]
            loc2, sound2 = events[i + 1]
            gap_sec = (loc2 - loc1) / 44100
            if sound1 == "S1" and sound2 == "S2":
                systolic_gaps.append(gap_sec)
            elif sound1 == "S2" and sound2 == "S1":
                diastolic_gaps.append(gap_sec)

        if systolic_gaps and diastolic_gaps:
            ground_truth[fname] = np.mean(systolic_gaps) / np.mean(diastolic_gaps)

    return ground_truth


def main():
    if not os.path.exists(TIMING_CSV_PATH):
        print(f"ERROR: '{TIMING_CSV_PATH}' not found.")
        return

    with open(TIMING_CSV_PATH, newline='') as f:
        rows = list(csv.reader(f))

    ground_truth = load_ground_truth_sd_ratio(rows)
    print(f"Computed ground-truth systolic/diastolic ratio for {len(ground_truth)} files.\n")

    if not os.path.isdir(NORMAL_AUDIO_DIR):
        print(f"ERROR: '{NORMAL_AUDIO_DIR}' not found.")
        return

    available_files = {f for f in os.listdir(NORMAL_AUDIO_DIR) if f.endswith(".wav")}

    errors = []
    print(f"{'File':<35}{'GroundTruth':>13}{'Estimated':>12}{'AbsError':>10}")
    print("-" * 70)

    for fname, gt_ratio in ground_truth.items():
        basename = os.path.basename(fname)
        if not basename.endswith(".wav"):
            basename += ".wav"
        if basename not in available_files:
            continue

        filepath = os.path.join(NORMAL_AUDIO_DIR, basename)
        result = extract_cardiac_parameters(filepath)
        est_ratio = result["systolic_diastolic_ratio"]

        if est_ratio is not None:
            err = abs(est_ratio - gt_ratio)
            errors.append(err)
            print(f"{basename:<35}{gt_ratio:>13.3f}{est_ratio:>12.3f}{err:>10.3f}")
        else:
            print(f"{basename:<35}{gt_ratio:>13.3f}{'N/A':>12}{'N/A':>10}")

    print("-" * 70)
    if errors:
        print(f"\nSystolic/Diastolic ratio -- Mean Absolute Error: {np.mean(errors):.3f} "
              f"(median {np.median(errors):.3f})")
        print(f"Matched {len(errors)} of {len(ground_truth)} ground-truth files.")
    else:
        print("No valid estimates produced.")


if __name__ == "__main__":
    main()
