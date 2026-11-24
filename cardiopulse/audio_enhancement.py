import numpy as np
import librosa

from spectrogram_utils import TARGET_SR, apply_bandpass, despike_glitches
from augmentation_utils import load_noise_pool

N_FFT_ENHANCE = 1024
HOP_LENGTH_ENHANCE = 256
SUBTRACTION_FACTOR = 1.5   # how aggressively to subtract the noise profile
SPECTRAL_FLOOR = 0.05      # keep at least this fraction of original magnitude,
                           # to avoid harsh "musical noise" artifacts from
                           # over-subtracting


from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from bpm_extraction import ENVELOPE_SMOOTH_SIGMA
from cardiac_parameters import detect_peaks_for_timing

GAP_MARGIN_SEC = 0.08  # excluded on either side of each peak, to avoid
                       # including actual heart-sound edges in the
                       # "noise-only" gap segments


def estimate_noise_profile_from_gaps(y, sr=TARGET_SR):
    """Estimates a noise profile from the quiet gaps BETWEEN heartbeats
    in the given audio -- no separate calibration step needed, since it
    works from audio already captured on-chest. Refreshes naturally each
    time it's called on a new buffer, so it adapts as you move between
    environments, unlike a one-time startup snapshot."""
    y_filtered = apply_bandpass(y, sr)
    analytic_signal = hilbert(y_filtered)
    envelope = gaussian_filter1d(np.abs(analytic_signal), sigma=ENVELOPE_SMOOTH_SIGMA)

    peaks = detect_peaks_for_timing(envelope, sr)
    if len(peaks) < 2:
        return None  # not enough beats detected to find gaps

    margin_samples = int(GAP_MARGIN_SEC * sr)
    gap_segments = []
    for i in range(len(peaks) - 1):
        gap_start = peaks[i] + margin_samples
        gap_end = peaks[i + 1] - margin_samples
        if gap_end > gap_start:
            gap_segments.append(y_filtered[gap_start:gap_end])

    if not gap_segments:
        return None

    gap_audio = np.concatenate(gap_segments)
    if len(gap_audio) < N_FFT_ENHANCE:
        return None  # too little gap audio to get a stable spectrum

    gap_stft = librosa.stft(gap_audio, n_fft=N_FFT_ENHANCE, hop_length=HOP_LENGTH_ENHANCE)
    return np.mean(np.abs(gap_stft), axis=1, keepdims=True)


def estimate_noise_profile(sr=TARGET_SR, y_for_gaps=None, blend_with_static=True):
    """Builds a noise magnitude spectrum profile, blending two sources:
    your recorded ESP32 noise clips (static, but consistent) and, when
    `y_for_gaps` is provided, a live estimate from the quiet gaps between
    heartbeats in that specific audio (adapts to the current environment
    automatically, no calibration step needed)."""
    static_profile = None
    noise_pool = load_noise_pool(target_sr=sr)
    if noise_pool is not None:
        noise_stft = librosa.stft(noise_pool, n_fft=N_FFT_ENHANCE, hop_length=HOP_LENGTH_ENHANCE)
        static_profile = np.mean(np.abs(noise_stft), axis=1, keepdims=True)

    gap_profile = None
    if y_for_gaps is not None:
        gap_profile = estimate_noise_profile_from_gaps(y_for_gaps, sr)

    if gap_profile is not None and static_profile is not None and blend_with_static:
        return (gap_profile + static_profile) / 2
    elif gap_profile is not None:
        return gap_profile
    else:
        return static_profile


def spectral_subtraction(y, sr=TARGET_SR, noise_profile=None):
    """Isolates the heart sound by subtracting the estimated noise
    fingerprint from the signal's frequency content, keeping the
    original phase. This is for PLAYBACK/listening only -- never fed to
    the model, since it's a different preprocessing path than training
    used and would shift the model's input distribution."""
    if noise_profile is None:
        noise_profile = estimate_noise_profile(sr, y_for_gaps=y)
    if noise_profile is None:
        return y  # no noise clips or usable gaps found -- return unmodified

    stft = librosa.stft(y, n_fft=N_FFT_ENHANCE, hop_length=HOP_LENGTH_ENHANCE)
    magnitude, phase = np.abs(stft), np.angle(stft)

    subtracted = magnitude - SUBTRACTION_FACTOR * noise_profile
    floored = np.maximum(subtracted, SPECTRAL_FLOOR * magnitude)

    enhanced_stft = floored * np.exp(1j * phase)
    enhanced = librosa.istft(enhanced_stft, hop_length=HOP_LENGTH_ENHANCE, length=len(y))
    return enhanced.astype(np.float32)


def get_playback_versions(raw_int16_samples, sr=TARGET_SR):
    """Given raw int16 samples straight off the ESP32, returns all three
    playback versions: raw, filtered (exactly what the model sees), and
    enhanced (noise removed using a blend of your static noise clips and
    a live estimate from the quiet gaps between heartbeats in THIS
    audio -- adapts to the current environment automatically, no
    calibration step needed, and only ever affects this one output)."""
    y_raw = np.array(raw_int16_samples, dtype=np.float32) / 32768.0

    y_despiked = despike_glitches(y_raw)
    y_filtered = apply_bandpass(y_despiked, sr)

    y_enhanced = spectral_subtraction(y_filtered, sr)

    return {
        "raw": y_raw,
        "filtered": y_filtered,
        "enhanced": y_enhanced,
    }
