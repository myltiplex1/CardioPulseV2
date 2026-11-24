import os
import numpy as np
import wave
from collections import deque, Counter

import model_inference
import cardiac_parameters
from spectrogram_utils import TARGET_SR, EXPECTED_SAMPLES

# --- CONFIGURATION ---
TEST_AUDIO_FILE = "test_audio_5min_normal.wav"
CLASSIFICATION_STEP_SEC = 2
CARDIAC_PARAMS_STEP_SEC = 15
CARDIAC_PARAMS_BUFFER_SEC = 15
SAVE_DIR = "saved_recordings"

# Majority-vote smoothing: instead of trusting any single 2s snapshot,
# require agreement across several recent classifications. Same
# "many independent readings, trust the consensus" principle that fixed
# BPM's octave-error instability -- classification confidence per chunk
# is inherently noisy on hard cases (Extra_Systole vs Normal especially,
# per the already-documented 0.31 precision), so smoothing over time is
# a legitimate mitigation, not a workaround.
VOTE_WINDOW_SIZE = 5  # ~10 seconds of history at CLASSIFICATION_STEP_SEC=2


def load_wav_as_int16(filepath):
    with wave.open(filepath, 'rb') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    samples = np.frombuffer(raw, dtype=np.int16)
    return samples, sr


def main():
    if not os.path.exists(TEST_AUDIO_FILE):
        print(f"ERROR: '{TEST_AUDIO_FILE}' not found. Run build_test_audio.py first.")
        return

    print(f"Loading '{TEST_AUDIO_FILE}'...")
    samples, sr = load_wav_as_int16(TEST_AUDIO_FILE)
    if sr != TARGET_SR:
        print(f"WARNING: file sample rate ({sr}) doesn't match expected {TARGET_SR}.")

    duration_sec = len(samples) / sr
    print(f"Loaded {duration_sec:.1f} seconds of audio.\n")

    print("Loading models (this may take a moment)...")
    model_inference.load_models()
    print()
    print("OFFLINE TEST MODE -- no live motion data available. "
          "This tests classification and cardiac parameters only.\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    classification_step_samples = int(CLASSIFICATION_STEP_SEC * sr)
    cardiac_step_samples = int(CARDIAC_PARAMS_STEP_SEC * sr)
    cardiac_buffer_samples = int(CARDIAC_PARAMS_BUFFER_SEC * sr)

    pos = EXPECTED_SAMPLES  # start once we have at least one full chunk
    last_cardiac_pos = 0
    latest_class, latest_confidence = "warming up...", None
    latest_cardiac = None
    vote_history = deque(maxlen=VOTE_WINDOW_SIZE)

    print("=" * 100)
    print(" OFFLINE PIPELINE TEST")
    print("=" * 100)

    while pos < len(samples):
        # --- Classification, every CLASSIFICATION_STEP_SEC of audio ---
        chunk = samples[pos - EXPECTED_SAMPLES:pos]
        latest_class, latest_confidence, _ = model_inference.classify_chunk(chunk)
        vote_history.append(latest_class)

        # Majority vote over recent history -- the more stable, actually
        # displayed result, since any single raw classification is noisy
        # on known-hard cases like Extra_Systole vs Normal.
        vote_counts = Counter(vote_history)
        smoothed_class, smoothed_count = vote_counts.most_common(1)[0]
        smoothed_agreement = smoothed_count / len(vote_history)

        # --- Cardiac parameters, every CARDIAC_PARAMS_STEP_SEC of audio ---
        if pos - last_cardiac_pos >= cardiac_step_samples and pos >= cardiac_buffer_samples:
            long_chunk = samples[pos - cardiac_buffer_samples:pos]
            y = long_chunk.astype(np.float32) / 32768.0
            tmp_path = os.path.join(SAVE_DIR, "_tmp_offline_test.wav")
            with wave.open(tmp_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(long_chunk.tobytes())
            latest_cardiac = cardiac_parameters.extract_cardiac_parameters(tmp_path)
            last_cardiac_pos = pos

        elapsed = pos / sr

        if latest_cardiac:
            bpm_str = f"{latest_cardiac['bpm']:.1f}" if latest_cardiac['bpm'] else "--"
            sd_str = f"{latest_cardiac['systolic_diastolic_ratio']:.3f}" if latest_cardiac['systolic_diastolic_ratio'] else "--"
            quality_str = f"{latest_cardiac['quality_score']:.0f}" if latest_cardiac['quality_score'] else "--"
            hrv_str = f"{latest_cardiac['hrv_sdnn_ms']:.1f}ms" if latest_cardiac['hrv_sdnn_ms'] else "--"
            amp_str = f"{latest_cardiac['s1_s2_amplitude_ratio']:.2f}" if latest_cardiac['s1_s2_amplitude_ratio'] else "--"
            reg_str = f"{latest_cardiac['rhythm_regularity_cv']:.3f}" if latest_cardiac['rhythm_regularity_cv'] else "--"
        else:
            bpm_str = sd_str = quality_str = hrv_str = amp_str = reg_str = "--"

        print(f"[{elapsed:6.1f}s] Raw: {latest_class:<13}({latest_confidence:.2f})  "
              f"Smoothed: {smoothed_class:<13}({smoothed_agreement:.0%} agree)  "
              f"BPM:{bpm_str:<6} S/D:{sd_str:<6} Q:{quality_str:<4} "
              f"HRV:{hrv_str:<8} Amp:{amp_str:<6} Reg:{reg_str}")

        pos += classification_step_samples

    print("=" * 70)
    print("Offline test complete.")
    if latest_cardiac:
        print("\nFinal cardiac parameters snapshot:")
        for k, v in latest_cardiac.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
