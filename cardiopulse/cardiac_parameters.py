import numpy as np
import librosa
from scipy.signal import find_peaks

from spectrogram_utils import TARGET_SR, apply_bandpass
from bpm_extraction import extract_envelope, estimate_bpm, MIN_PLAUSIBLE_BPM, MAX_PLAUSIBLE_BPM


def detect_peaks_for_timing(envelope, sr=TARGET_SR):
    """Detects individual S1/S2-candidate peaks across the whole envelope.
    Distinct from bpm_extraction's windowed rate estimate, which only
    needs A RATE -- this needs actual peak positions and amplitudes for
    interval/timing analysis."""
    min_distance_samples = int(sr * 60 / MAX_PLAUSIBLE_BPM / 2.5)
    peak_height = np.mean(envelope) + 0.5 * np.std(envelope)
    peaks, _ = find_peaks(envelope, distance=min_distance_samples, height=peak_height)
    return peaks


def classify_systole_diastole(peak_times_sec):
    """Splits consecutive peak-to-peak intervals into systolic (shorter,
    S1-to-S2) and diastolic (longer, S2-to-next-S1) groups by alternation.
    Assumption: systole is shorter than diastole at rest-to-moderate heart
    rates -- a standard physiological pattern, though it can break down
    at very high heart rates or atypical recordings where diastole
    shortens below systole. This is a documented, structural limitation:
    windowing (below) improves stability within a recording, but cannot
    fix a recording where this assumption doesn't hold at all -- that
    needs genuine S1-vs-S2 identification, not implemented here."""
    if len(peak_times_sec) < 3:
        return None

    intervals = np.diff(peak_times_sec)
    even_intervals = intervals[0::2]
    odd_intervals = intervals[1::2]

    if len(even_intervals) == 0 or len(odd_intervals) == 0:
        return None

    if np.mean(even_intervals) <= np.mean(odd_intervals):
        systolic_intervals = even_intervals
        diastolic_intervals = odd_intervals
        systolic_start_indices = list(range(0, len(peak_times_sec) - 1, 2))
    else:
        systolic_intervals = odd_intervals
        diastolic_intervals = even_intervals
        systolic_start_indices = list(range(1, len(peak_times_sec) - 1, 2))

    return {
        "systolic_intervals": systolic_intervals,
        "diastolic_intervals": diastolic_intervals,
        "systolic_start_indices": systolic_start_indices,
    }


def windowed_systolic_diastolic_ratio(envelope, sr=TARGET_SR, window_sec=3, hop_sec=1.5):
    """Computes the systolic/diastolic ratio over several overlapping
    windows and takes the median -- the same robustness approach that
    fixed BPM's octave-error instability. A single window's peak
    detection going astray (as happened on one of the validation files)
    gets outvoted by the majority landing on a stable estimate. Falls
    back to a single whole-signal pass if the recording is too short."""
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    def _single_pass_ratio(env_segment):
        peaks = detect_peaks_for_timing(env_segment, sr)
        peak_times = peaks / sr
        split = classify_systole_diastole(peak_times)
        if split is None:
            return None
        mean_sys = np.mean(split["systolic_intervals"])
        mean_dia = np.mean(split["diastolic_intervals"])
        return mean_sys / mean_dia if mean_dia > 0 else None

    if len(envelope) < window_samples:
        ratio = _single_pass_ratio(envelope)
        return ratio

    estimates = []
    start = 0
    while start + window_samples <= len(envelope):
        ratio = _single_pass_ratio(envelope[start:start + window_samples])
        if ratio is not None:
            estimates.append(ratio)
        start += hop_samples

    if not estimates:
        return None
    return float(np.median(estimates))


def extract_cardiac_parameters(filepath, sr=TARGET_SR):
    """Extracts BPM (via the already-validated bpm_extraction pipeline)
    plus additional cardiac parameters: HRV, systolic/diastolic ratio,
    S1/S2 amplitude ratio, rhythm regularity, and a recording quality
    score. Only BPM has been validated against ground truth so far --
    the rest are new and should be treated as provisional until checked
    (see validate_sd_ratio.py for the systolic/diastolic split,
    which CAN be validated using set_a_timing.csv's S1/S2 labels)."""
    y, _ = librosa.load(filepath, sr=sr)
    envelope = extract_envelope(y, sr)

    bpm_result = estimate_bpm(filepath, sr)

    peak_indices = detect_peaks_for_timing(envelope, sr)
    peak_times_sec = peak_indices / sr
    peak_amplitudes = envelope[peak_indices]

    result = {
        "bpm": bpm_result["bpm_final"],
        "bpm_confidence": bpm_result["confidence"],
        "hrv_sdnn_ms": None,
        "rhythm_regularity_cv": None,   # coefficient of variation of beat intervals
        "systolic_diastolic_ratio": None,
        "systolic_diastolic_ratio_whole_signal": None,
        "s1_s2_amplitude_ratio": None,
        "quality_score": None,
        "n_peaks_detected": len(peak_indices),
        "envelope": envelope,
        "peak_indices": peak_indices,
    }

    if len(peak_times_sec) >= 3:
        sd_split = classify_systole_diastole(peak_times_sec)
        if sd_split is not None:
            n_pairs = min(len(sd_split["systolic_intervals"]), len(sd_split["diastolic_intervals"]))
            if n_pairs >= 2:
                full_cycle_sec = (sd_split["systolic_intervals"][:n_pairs] +
                                  sd_split["diastolic_intervals"][:n_pairs])
                full_cycle_ms = full_cycle_sec * 1000
                result["hrv_sdnn_ms"] = float(np.std(full_cycle_ms))
                result["rhythm_regularity_cv"] = float(np.std(full_cycle_ms) / np.mean(full_cycle_ms))

            mean_systolic = np.mean(sd_split["systolic_intervals"])
            mean_diastolic = np.mean(sd_split["diastolic_intervals"])
            if mean_diastolic > 0:
                # Whole-signal ratio, for reference/comparison
                whole_signal_ratio = mean_systolic / mean_diastolic
            else:
                whole_signal_ratio = None

            # Windowed, median-aggregated ratio -- the more robust estimate,
            # same principle as the BPM windowing fix. This is the one
            # reported as the final systolic_diastolic_ratio.
            result["systolic_diastolic_ratio"] = windowed_systolic_diastolic_ratio(envelope, sr)
            result["systolic_diastolic_ratio_whole_signal"] = whole_signal_ratio

            s1_indices = sd_split["systolic_start_indices"]
            s1_amplitudes = [peak_amplitudes[i] for i in s1_indices if i < len(peak_amplitudes)]
            s2_indices = [i + 1 for i in s1_indices if i + 1 < len(peak_amplitudes)]
            s2_amplitudes = [peak_amplitudes[i] for i in s2_indices]
            if s1_amplitudes and s2_amplitudes and np.mean(s2_amplitudes) > 0:
                result["s1_s2_amplitude_ratio"] = float(np.mean(s1_amplitudes) / np.mean(s2_amplitudes))

    # --- Recording quality score (0-100), combining BPM confidence with
    # the same peakiness/loudness diagnostics used earlier to explain the
    # Set A / Set B gap ---
    y_filtered = apply_bandpass(y, sr)
    peakiness = np.std(envelope) / (np.mean(envelope) + 1e-10)

    confidence_score_map = {"high": 100, "medium": 60, "low": 20}
    conf_component = confidence_score_map.get(bpm_result["confidence"], 20)
    peakiness_component = min(100, peakiness * 40)  # rough scaling: peakiness ~2.5 -> 100
    result["quality_score"] = float(np.clip(0.6 * conf_component + 0.4 * peakiness_component, 0, 100))

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        print("Usage: python cardiac_parameters.py <path_to_wav_file>")
        sys.exit(1)

    result = extract_cardiac_parameters(filepath)
    print(f"File: {filepath}")
    for k, v in result.items():
        print(f"  {k}: {v}")
