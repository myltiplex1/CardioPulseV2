import os
import time
import wave
import numpy as np

from esp32_streams import AudioReceiver, MotionReceiver
import model_inference
import cardiac_parameters
import audio_enhancement
from spectrogram_utils import TARGET_SR, EXPECTED_SAMPLES

# --- CONFIGURATION ---
ESP32_IP = "192.168.4.1"
TCP_PORT_AUDIO = 3333
TCP_PORT_MOTION = 3334

CLASSIFICATION_INTERVAL_SEC = 2      # how often to classify a fresh 3s chunk
CARDIAC_PARAMS_INTERVAL_SEC = 15     # how often to run the longer analysis
CARDIAC_PARAMS_BUFFER_SEC = 15       # how much audio to use for it

SAVE_DIR = "saved_recordings"


def save_wav(filepath, float_audio, sr=TARGET_SR):
    int16_audio = np.clip(float_audio * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_audio.tobytes())


def save_playback_versions(raw_samples, label, sr=TARGET_SR):
    """Saves raw/filtered/enhanced versions of a segment locally so they
    can be played back (e.g. via any media player) -- generating the
    audio itself, not a playback UI, which belongs in the later
    dashboard."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    versions = audio_enhancement.get_playback_versions(raw_samples, sr)

    paths = {}
    for name, audio in versions.items():
        path = os.path.join(SAVE_DIR, f"{label}_{name}.wav")
        save_wav(path, audio, sr)
        paths[name] = path

    return paths


def main():
    print("Connecting to ESP32...")
    audio = AudioReceiver(ESP32_IP, TCP_PORT_AUDIO)
    motion = MotionReceiver(ESP32_IP, TCP_PORT_MOTION)
    audio.start()
    motion.start()
    print("Connected.\n")

    print("Loading models (this may take a moment)...")
    model_inference.load_models()
    print("Noise removal for playback adapts automatically from the gaps "
          "between heartbeats in each recording -- no calibration step needed.\n")

    input("Press ENTER to start live monitoring (Ctrl+C to stop)...")
    start_time = time.time()
    audio.start_recording()
    motion.mark_recording_start(start_time)

    last_classification_time = 0
    last_cardiac_time = 0
    latest_class = "warming up..."
    latest_confidence = None
    latest_cardiac = None
    latest_motion_reliability = "unknown"

    try:
        while True:
            now = time.time()

            # --- Path A: short-window classification ---
            if now - last_classification_time >= CLASSIFICATION_INTERVAL_SEC:
                recent = audio.get_recent_samples(EXPECTED_SAMPLES)
                if recent is not None:
                    latest_class, latest_confidence, _ = model_inference.classify_chunk(recent)
                last_classification_time = now

            # --- Path B: longer-window cardiac parameters, WITH motion
            # tracked over that exact same window (motion-artifact flagging) ---
            if now - last_cardiac_time >= CARDIAC_PARAMS_INTERVAL_SEC:
                n_samples_needed = int(CARDIAC_PARAMS_BUFFER_SEC * TARGET_SR)
                recent_long = audio.get_recent_samples(n_samples_needed)
                if recent_long is not None:
                    window_start = now - CARDIAC_PARAMS_BUFFER_SEC
                    y = np.array(recent_long, dtype=np.float32) / 32768.0
                    tmp_path = os.path.join(SAVE_DIR, "_tmp_cardiac_buffer.wav")
                    os.makedirs(SAVE_DIR, exist_ok=True)
                    save_wav(tmp_path, y)
                    latest_cardiac = cardiac_parameters.extract_cardiac_parameters(tmp_path)

                    # Motion-artifact flagging: how much motion happened
                    # during THIS SPECIFIC window the cardiac reading came from
                    window_motion_scores = motion.get_scores_in_window(window_start, now)
                    if window_motion_scores:
                        high_frac = sum(1 for s in window_motion_scores if s >= 0.10) / len(window_motion_scores)
                        if high_frac > 0.10:
                            latest_motion_reliability = "LOW (motion detected during this reading)"
                        elif high_frac > 0:
                            latest_motion_reliability = "moderate (some motion detected)"
                        else:
                            latest_motion_reliability = "good (minimal motion during capture)"
                    else:
                        latest_motion_reliability = "unknown"
                last_cardiac_time = now

            motion_score, motion_tier = motion.get_current_status()
            activity_avg, activity_label = motion.get_activity_context()

            # --- Terminal display ---
            conf_str = f"{latest_confidence:.2f}" if latest_confidence else "--"

            if latest_cardiac:
                bpm_str = f"{latest_cardiac['bpm']:.1f}" if latest_cardiac['bpm'] else "--"
                sd_ratio_str = (f"{latest_cardiac['systolic_diastolic_ratio']:.3f}"
                                if latest_cardiac['systolic_diastolic_ratio'] else "--")
                quality_str = f"{latest_cardiac['quality_score']:.0f}" if latest_cardiac['quality_score'] else "--"
                hrv_str = f"{latest_cardiac['hrv_sdnn_ms']:.1f}ms" if latest_cardiac['hrv_sdnn_ms'] else "--"
                amp_ratio_str = (f"{latest_cardiac['s1_s2_amplitude_ratio']:.2f}"
                                 if latest_cardiac['s1_s2_amplitude_ratio'] else "--")
                regularity_str = (f"{latest_cardiac['rhythm_regularity_cv']:.3f}"
                                  if latest_cardiac['rhythm_regularity_cv'] else "--")
            else:
                bpm_str = sd_ratio_str = quality_str = hrv_str = amp_ratio_str = regularity_str = "--"

            print("\033[H\033[J", end='')  # clear terminal for a clean redraw each update

            audio_connected = audio.is_actively_receiving()
            motion_connected = motion.is_actively_receiving()
            device_connected = audio_connected and motion_connected
            status_line = "CardioPulse connected" if device_connected else "Not connected"

            print("=" * 70)
            print(f" Hi Enoch -- {status_line}")
            print("=" * 70)
            print(f"Classification:  {latest_class}  (confidence: {conf_str})")
            print(f"Motion (now):    {motion_tier}  (score: {motion_score:.3f})")
            print(f"Activity context (last 3 min): {activity_label}"
                  + (f" (avg score: {activity_avg:.3f})" if activity_avg is not None else ""))
            print()
            print("--- Cardiac Parameters: VALIDATED ---")
            print(f"  BPM:                    {bpm_str}")
            print(f"  Systolic/Diastolic:     {sd_ratio_str}")
            print(f"  Quality score:          {quality_str}/100")
            print(f"  Reading reliability:    {latest_motion_reliability}")
            print()
            print("--- Cardiac Parameters: EXPERIMENTAL (not yet validated) ---")
            print(f"  HRV (SDNN):             {hrv_str}")
            print(f"  S1/S2 amplitude ratio:  {amp_ratio_str}")
            print(f"  Rhythm regularity:      {regularity_str}")
            print("=" * 70)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    audio.stop_recording()
    time.sleep(0.3)
    audio.close()
    motion.close()

    # --- Offer to save a playback snippet from what's in the buffer ---
    save_choice = input("\nSave the last few seconds for playback (raw/filtered/enhanced)? [y/N]: ")
    if save_choice.lower() == 'y':
        recent = audio.get_recent_samples(int(10 * TARGET_SR))
        if recent is not None:
            label = f"session_{int(time.time())}"
            paths = save_playback_versions(recent, label)
            print("Saved:")
            for name, path in paths.items():
                print(f"  {name}: {path}")
        else:
            print("Not enough audio buffered to save.")


if __name__ == "__main__":
    main()
