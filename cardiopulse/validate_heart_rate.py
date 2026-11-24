import os
import csv
import numpy as np

from bpm_extraction import estimate_bpm

# --- CONFIGURATION ---
TIMING_CSV_PATH = "set_a_timing.csv"       # adjust path if it's elsewhere
NORMAL_AUDIO_DIR = "Organized_Heart_Audio/Normal"


def inspect_csv(csv_path):
    """Prints the actual structure of the timing CSV before assuming
    anything about its columns -- the exact schema wasn't confirmed."""
    if not os.path.exists(csv_path):
        print(f"ERROR: '{csv_path}' not found. Update TIMING_CSV_PATH.")
        return None

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    print("=" * 60)
    print(" set_a_timing.csv -- STRUCTURE INSPECTION")
    print("=" * 60)
    print(f"Total rows (including header): {len(rows)}")
    print(f"Header/first row: {rows[0]}")
    print("First 10 data rows:")
    for row in rows[1:11]:
        print(f"  {row}")
    print("=" * 60)
    return rows


def compute_ground_truth_bpm(rows):
    """Attempts to compute ground-truth BPM per file from the timing
    annotations. ASSUMES columns are something like
    [fname, cycle, sound, location] where 'location' is a sample index
    and 'sound' marks S1/S2 -- adjust this function if the printed
    structure above shows a different layout."""
    header = rows[0]
    data = rows[1:]

    print("\nAttempting to parse with assumed columns:", header)
    print("If this looks wrong based on the inspection above, this "
          "function needs adjusting to match the real column names.\n")

    file_s1_locations = {}

    try:
        fname_idx = header.index("fname")
        sound_idx = header.index("sound")
        location_idx = header.index("location")
    except ValueError:
        print("Could not find expected column names ('fname', 'sound', "
              "'location') in the header. Please check the inspection "
              "output above and adjust this script's column handling.")
        return {}

    for row in data:
        if len(row) <= max(fname_idx, sound_idx, location_idx):
            continue
        fname = row[fname_idx]
        sound = row[sound_idx]
        try:
            location = int(row[location_idx])
        except (ValueError, IndexError):
            continue

        if sound == "S1":
            file_s1_locations.setdefault(fname, []).append(location)

    ground_truth_bpm = {}
    for fname, locations in file_s1_locations.items():
        if len(locations) < 2:
            continue
        locations = sorted(locations)
        intervals_samples = np.diff(locations)
        # These CSVs are typically indexed at the audio's native sample rate;
        # if BPM numbers come out implausible, the sample rate assumption
        # below may need adjusting to match the actual file's native rate.
        avg_interval_sec = np.mean(intervals_samples) / 44100
        if avg_interval_sec > 0:
            ground_truth_bpm[fname] = 60.0 / avg_interval_sec

    return ground_truth_bpm


def main():
    rows = inspect_csv(TIMING_CSV_PATH)
    if rows is None:
        return

    ground_truth = compute_ground_truth_bpm(rows)
    if not ground_truth:
        print("No ground-truth BPM values could be computed -- likely "
              "the column assumption above didn't match. Check the "
              "inspection output and adjust compute_ground_truth_bpm().")
        return

    print(f"Computed ground-truth BPM for {len(ground_truth)} files.\n")

    if not os.path.isdir(NORMAL_AUDIO_DIR):
        print(f"ERROR: '{NORMAL_AUDIO_DIR}' not found.")
        return

    available_files = {f for f in os.listdir(NORMAL_AUDIO_DIR) if f.endswith(".wav")}

    errors_autocorr = []
    errors_peaks = []
    errors_final = []
    high_confidence_count = 0
    matched = 0

    print("=" * 100)
    print(f"{'File':<32}{'GroundTruth':>12}{'Autocorr':>11}{'PeakCount':>11}{'Final':>9}{'WinAgree':>10}{'Confidence':>12}")
    print("-" * 100)

    for fname, gt_bpm in ground_truth.items():
        basename = os.path.basename(fname)
        if not basename.endswith(".wav"):
            basename += ".wav"

        if basename not in available_files:
            continue

        matched += 1
        filepath = os.path.join(NORMAL_AUDIO_DIR, basename)
        result = estimate_bpm(filepath)

        auto_bpm = result["bpm_autocorrelation"]
        peak_bpm = result["bpm_peak_counting"]
        final_bpm = result["bpm_final"]
        confidence = result["confidence"]
        win_agree = result["window_agreement_fraction"]

        auto_str = f"{auto_bpm:.1f}" if auto_bpm else "N/A"
        peak_str = f"{peak_bpm:.1f}" if peak_bpm else "N/A"
        final_str = f"{final_bpm:.1f}" if final_bpm else "N/A"
        win_str = f"{win_agree:.2f}" if win_agree is not None else "N/A"
        print(f"{basename:<32}{gt_bpm:>12.1f}{auto_str:>11}{peak_str:>11}{final_str:>9}{win_str:>10}{confidence:>12}")

        if auto_bpm:
            errors_autocorr.append(abs(auto_bpm - gt_bpm))
        if peak_bpm:
            errors_peaks.append(abs(peak_bpm - gt_bpm))
        if final_bpm:
            errors_final.append(abs(final_bpm - gt_bpm))
        if confidence == "high":
            high_confidence_count += 1

    print("-" * 100)
    print(f"\nMatched {matched} of {len(ground_truth)} ground-truth files to local audio.")

    if errors_autocorr:
        print(f"\nAutocorrelation method -- Mean Absolute Error: "
              f"{np.mean(errors_autocorr):.2f} BPM "
              f"(median {np.median(errors_autocorr):.2f})")
    if errors_peaks:
        print(f"Peak-counting method   -- Mean Absolute Error: "
              f"{np.mean(errors_peaks):.2f} BPM "
              f"(median {np.median(errors_peaks):.2f})")
    if errors_final:
        print(f"Final (agreement-aware) -- Mean Absolute Error: "
              f"{np.mean(errors_final):.2f} BPM "
              f"(median {np.median(errors_final):.2f})")
    print(f"\nHigh-confidence readings: {high_confidence_count}/{matched} "
          f"({100*high_confidence_count/matched:.0f}%)")


if __name__ == "__main__":
    main()
