import numpy as np
from scipy.signal import hilbert, find_peaks
from scipy.ndimage import gaussian_filter1d

from spectrogram_utils import TARGET_SR, apply_bandpass

# --- CONFIGURATION ---
ENVELOPE_SMOOTH_SIGMA = 100     # ~6ms smoothing window at 16kHz
MIN_PLAUSIBLE_BPM = 40          # physiological bounds for a human heart rate
MAX_PLAUSIBLE_BPM = 200


def enhance_envelope_contrast(envelope, sr, baseline_window_sec=1.0):
    """Subtracts a slowly-varying local baseline from the envelope, so
    genuine S1/S2 peaks stand out more clearly against whatever
    continuous background level sits underneath them. Targets the
    specific issue found in diagnosis: some recordings (notably Set B)
    are louder overall but less 'peaky' -- the heart sounds don't stand
    out distinctly from the background, rather than simply being quiet.
    On already-peaky recordings this has minimal effect, since their
    baseline is already close to flat/zero."""
    from scipy.ndimage import median_filter
    baseline_window_samples = int(baseline_window_sec * sr)
    if baseline_window_samples % 2 == 0:
        baseline_window_samples += 1  # median_filter prefers an odd window
    baseline = median_filter(envelope, size=baseline_window_samples)
    enhanced = envelope - baseline
    return np.clip(enhanced, 0, None)  # envelope amplitude can't go negative


def extract_envelope(y, sr=TARGET_SR):
    """Band-pass filters the signal, then extracts a smoothed amplitude
    envelope via the Hilbert transform -- the same approach already used
    in visualize_envelopes.py, reused here as the basis for beat detection.
    Contrast enhancement is applied afterward to help recordings where
    heart sounds don't stand out clearly from the background level."""
    y_filtered = apply_bandpass(y, sr)
    analytic_signal = hilbert(y_filtered)
    envelope = np.abs(analytic_signal)
    envelope = gaussian_filter1d(envelope, sigma=ENVELOPE_SMOOTH_SIGMA)
    return enhance_envelope_contrast(envelope, sr)


def _autocorrelation_period_bpm(envelope, sr):
    """Single-shot autocorrelation period estimate (simple global max,
    no heuristic correction) -- used as the building block for the
    more robust windowed version below, which handles octave errors
    by outvoting rather than by a fragile single-shot heuristic."""
    envelope = envelope - np.mean(envelope)
    autocorr = np.correlate(envelope, envelope, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]

    min_lag = int(sr * 60 / MAX_PLAUSIBLE_BPM)
    max_lag = int(sr * 60 / MIN_PLAUSIBLE_BPM)
    max_lag = min(max_lag, len(autocorr) - 1)

    if min_lag >= max_lag:
        return None

    search_region = autocorr[min_lag:max_lag]
    if len(search_region) == 0 or np.max(search_region) <= 0:
        return None

    best_lag = min_lag + np.argmax(search_region)
    return 60.0 / (best_lag / sr)


def estimate_bpm_autocorrelation(envelope, sr=TARGET_SR, window_sec=3, hop_sec=1.5):
    """Estimates heart rate via autocorrelation, computed over several
    overlapping shorter windows rather than the whole recording at once.
    Returns (median_bpm, list_of_per_window_estimates) -- the per-window
    list lets the caller measure how internally consistent this estimate
    is, which is a more informative confidence signal than comparing
    against a separate, weaker method alone."""
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    if len(envelope) < window_samples:
        single = _autocorrelation_period_bpm(envelope, sr)
        return (single, [single] if single is not None else [])

    estimates = []
    start = 0
    while start + window_samples <= len(envelope):
        window = envelope[start:start + window_samples]
        bpm = _autocorrelation_period_bpm(window, sr)
        if bpm is not None:
            estimates.append(bpm)
        start += hop_samples

    if not estimates:
        return (None, [])
    return (float(np.median(estimates)), estimates)


def estimate_bpm_peak_counting(envelope, sr=TARGET_SR):
    """Estimates heart rate by detecting individual envelope peaks (S1
    and S2 together), then assuming roughly 2 dominant peaks per cardiac
    cycle. Less robust than autocorrelation for noisy/abnormal recordings,
    but useful as an independent cross-check."""
    min_distance_samples = int(sr * 60 / MAX_PLAUSIBLE_BPM / 2.5)
    # divide by 2.5 rather than 2 -- S1-S2 spacing within one cycle is
    # shorter than the full cycle spacing, so peaks can legitimately be
    # closer together than one full beat period

    peak_height = np.mean(envelope) + 0.5 * np.std(envelope)
    peaks, _ = find_peaks(envelope, distance=min_distance_samples, height=peak_height)

    if len(peaks) < 2:
        return None

    duration_sec = len(envelope) / sr
    # Assume ~2 peaks (S1 + S2) per cardiac cycle
    n_cycles = len(peaks) / 2
    bpm = (n_cycles / duration_sec) * 60.0
    return bpm, peaks


def estimate_bpm(filepath, sr=TARGET_SR):
    """Loads a file and returns both BPM estimates plus the envelope,
    for inspection/plotting."""
    import librosa
    y, _ = librosa.load(filepath, sr=sr)
    envelope = extract_envelope(y, sr)

    bpm_autocorr, window_estimates = estimate_bpm_autocorrelation(envelope, sr)
    peak_result = estimate_bpm_peak_counting(envelope, sr)
    bpm_peaks, peak_indices = peak_result if peak_result else (None, None)

    # Primary confidence signal: how tightly do the individual windows
    # (already computed above) agree with each other? This measures
    # whether THIS recording itself gave a stable, consistent estimate --
    # independent of the weaker peak-counting method entirely.
    WINDOW_AGREEMENT_TOLERANCE_BPM = 10
    if bpm_autocorr is not None and len(window_estimates) >= 2:
        within_tolerance = sum(1 for e in window_estimates
                               if abs(e - bpm_autocorr) <= WINDOW_AGREEMENT_TOLERANCE_BPM)
        window_agreement_fraction = within_tolerance / len(window_estimates)
    else:
        window_agreement_fraction = None

    # Secondary, softer signal: does the weaker peak-counting method
    # corroborate the result? Used only to distinguish "high" from
    # "medium" confidence -- never strong enough alone to override a
    # stable autocorrelation estimate, since validation showed
    # peak-counting is the less accurate of the two methods.
    CROSS_CHECK_TOLERANCE_BPM = 15
    if bpm_autocorr is not None and bpm_peaks is not None:
        peak_agrees = abs(bpm_autocorr - bpm_peaks) <= CROSS_CHECK_TOLERANCE_BPM
    else:
        peak_agrees = None

    # Autocorrelation is the primary, more accurate method (validated
    # MAE ~5 BPM vs. peak-counting's ~10 BPM) -- always reported as the
    # final estimate rather than averaged with the weaker method, which
    # can only pull an already-good estimate off target.
    bpm_final = bpm_autocorr if bpm_autocorr is not None else bpm_peaks

    # Confidence tiering:
    #   high   -- windows internally consistent AND peak-counting agrees
    #   medium -- windows internally consistent but peak-counting disagrees
    #             (autocorrelation is still trusted; disagreement is just noted)
    #   low    -- the windows themselves disagree (this specific recording
    #             gave an unstable signal, regardless of cross-checking)
    #
    # Fallback (recording too short to compute window agreement at all):
    #   rely on peak-counting agreement alone rather than defaulting to
    #   low -- a short recording's single estimate can still be accurate,
    #   it just can't be internally cross-checked the same way.
    if bpm_final is None:
        confidence = "low"
    elif window_agreement_fraction is not None:
        if window_agreement_fraction >= 0.7:
            confidence = "high" if peak_agrees else "medium"
        else:
            confidence = "low"
    else:
        # too short to window -- fall back to peak-counting agreement only
        confidence = "medium" if peak_agrees else "low"

    return {
        "bpm_autocorrelation": bpm_autocorr,
        "bpm_peak_counting": bpm_peaks,
        "bpm_final": bpm_final,
        "confidence": confidence,
        "window_agreement_fraction": window_agreement_fraction,
        "methods_agree": peak_agrees,
        "envelope": envelope,
        "peak_indices": peak_indices,
        "audio": y,
        "sr": sr,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        print("Usage: python bpm_extraction.py <path_to_wav_file>")
        sys.exit(1)

    result = estimate_bpm(filepath)
    print(f"File: {filepath}")
    print(f"  Autocorrelation BPM estimate: {result['bpm_autocorrelation']}")
    print(f"  Peak-counting BPM estimate:   {result['bpm_peak_counting']}")
    print(f"  Final BPM:                    {result['bpm_final']}")
    print(f"  Window agreement: {result['window_agreement_fraction']}")
    print(f"  Confidence: {result['confidence']} (peak-counting agrees: {result['methods_agree']})")
