import socket
import struct
import threading
import time
import wave
from collections import deque

# --- CONFIGURATION ---
ESP32_IP = "192.168.4.1"
TCP_PORT_AUDIO = 3333
TCP_PORT_MOTION = 3334
SAMPLE_RATE = 16000
MOTION_SAMPLE_RATE_HZ = 50
OUTPUT_FILENAME = "combined_recording.wav"
MAX_RECORD_SECONDS = 60

ACCEL_SCALE = 16384.0
MOTION_WINDOW_SECONDS = 2
MOTION_WINDOW_SIZE = int(MOTION_SAMPLE_RATE_HZ * MOTION_WINDOW_SECONDS)

# 3-tier motion classification, set from real observed data:
# still avg ~0.004, walking avg ~0.055, walking peak ~0.18.
# Provisional starting point from 2 data points -- expect to refine with
# more real-world testing, same as everything else in this project.
MOTION_TIER_LOW_MAX = 0.02       # below this: still / minimal motion
MOTION_TIER_MODERATE_MAX = 0.10  # below this: moderate (e.g. walking average)
                                  # at/above this: high (e.g. walking footfall peaks)


def classify_motion_tier(score):
    if score < MOTION_TIER_LOW_MAX:
        return "low"
    elif score < MOTION_TIER_MODERATE_MAX:
        return "moderate"
    else:
        return "high"


class AudioReceiver(threading.Thread):
    def __init__(self, ip, port):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))
        self.sock.settimeout(0.5)
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.running = False
        self.stop_flag = threading.Event()

    def start_recording(self):
        self.sock.sendall(b'r')
        self.running = True

    def stop_recording(self):
        self.sock.sendall(b's')
        self.running = False

    def run(self):
        while not self.stop_flag.is_set():
            if not self.running:
                time.sleep(0.05)
                continue
            try:
                chunk = self.sock.recv(4096)
                if chunk:
                    with self.lock:
                        self.buffer.extend(chunk)
            except socket.timeout:
                continue
            except OSError:
                # Socket closed by close() from the main thread -- expected
                # during shutdown, not a real error.
                break

    def get_buffer_copy(self):
        with self.lock:
            return bytes(self.buffer)

    def close(self):
        self.stop_flag.set()
        self.sock.close()


class MotionReceiver(threading.Thread):
    def __init__(self, ip, port):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))
        self.sock.settimeout(0.5)
        self.magnitude_window = deque(maxlen=MOTION_WINDOW_SIZE)
        self.timestamped_scores = []  # (elapsed_seconds, motion_score) pairs
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.recording_start_time = None

    def mark_recording_start(self, start_time):
        self.recording_start_time = start_time

    def run(self):
        while not self.stop_flag.is_set():
            try:
                data = self.sock.recv(6)
            except socket.timeout:
                continue
            except OSError:
                # Socket was closed (likely by close() from the main
                # thread) while this thread was blocked in recv() --
                # expected during shutdown, not a real error.
                break
            if self.stop_flag.is_set():
                break
            if len(data) < 6:
                continue

            ax_raw, ay_raw, az_raw = struct.unpack('<hhh', data)
            ax, ay, az = ax_raw / ACCEL_SCALE, ay_raw / ACCEL_SCALE, az_raw / ACCEL_SCALE
            magnitude = (ax ** 2 + ay ** 2 + az ** 2) ** 0.5

            with self.lock:
                self.magnitude_window.append(magnitude)
                if len(self.magnitude_window) >= 2:
                    values = list(self.magnitude_window)
                    mean = sum(values) / len(values)
                    variance = sum((v - mean) ** 2 for v in values) / len(values)
                    score = variance ** 0.5

                    if self.recording_start_time is not None:
                        elapsed = time.time() - self.recording_start_time
                        self.timestamped_scores.append((elapsed, score))

    def get_scores_during(self, duration_seconds):
        with self.lock:
            return [s for t, s in self.timestamped_scores if 0 <= t <= duration_seconds]

    def close(self):
        self.stop_flag.set()
        self.sock.close()


def main():
    print(f"Connecting to audio ({TCP_PORT_AUDIO}) and motion ({TCP_PORT_MOTION})...")
    try:
        audio = AudioReceiver(ESP32_IP, TCP_PORT_AUDIO)
        motion = MotionReceiver(ESP32_IP, TCP_PORT_MOTION)
    except Exception as e:
        print(f"Error connecting: {e}")
        print("Check cardiopulse_firmware.ino is running and both devices "
              "are on the same WiFi network.")
        return

    audio.start()
    motion.start()
    print("Both connections established.\n")

    input(f"Press ENTER to start recording '{OUTPUT_FILENAME}' "
          f"(max {MAX_RECORD_SECONDS}s)...")

    start_time = time.time()
    audio.start_recording()
    motion.mark_recording_start(start_time)
    print("\n[ RECORDING IN PROGRESS... ] Press Ctrl+C to stop early.\n")

    try:
        while time.time() - start_time < MAX_RECORD_SECONDS:
            time.sleep(0.5)
        print("\n[ Max duration reached, stopping automatically ]")
    except KeyboardInterrupt:
        print("\n[ STOPPING... ]")

    elapsed = time.time() - start_time
    audio.stop_recording()
    time.sleep(0.3)

    raw_audio = audio.get_buffer_copy()
    safe_length = len(raw_audio) - (len(raw_audio) % 2)
    safe_audio = raw_audio[:safe_length]

    audio.close()
    motion.close()

    # --- Audio diagnostic (same as before) ---
    actual_samples = len(safe_audio) // 2
    expected_samples = int(elapsed * SAMPLE_RATE)
    dropout_pct = 100 * (1 - actual_samples / expected_samples) if expected_samples > 0 else 0

    # --- Motion-during-recording diagnostic (3-tier breakdown) ---
    motion_scores = motion.get_scores_during(elapsed)
    if motion_scores:
        avg_motion = sum(motion_scores) / len(motion_scores)
        max_motion = max(motion_scores)
        tiers = [classify_motion_tier(s) for s in motion_scores]
        pct_low = 100 * tiers.count("low") / len(tiers)
        pct_moderate = 100 * tiers.count("moderate") / len(tiers)
        pct_high = 100 * tiers.count("high") / len(tiers)
    else:
        avg_motion = max_motion = pct_low = pct_moderate = pct_high = None

    print("\n" + "=" * 60)
    print(" COMBINED RECORDING DIAGNOSTIC")
    print("=" * 60)
    print(f"Duration: {elapsed:.2f}s")
    print(f"Audio dropout: {dropout_pct:+.2f}%")
    if motion_scores:
        print(f"Motion during recording -- avg score: {avg_motion:.4f}, max score: {max_motion:.4f}")
        print(f"Motion tier breakdown: low={pct_low:.1f}%  moderate={pct_moderate:.1f}%  high={pct_high:.1f}%")
        if pct_high > 10:
            print("WARNING: significant portions of high motion detected -- "
                  "audio quality likely degraded during those moments.")
        elif pct_moderate > 40:
            print("NOTE: substantial moderate motion detected (e.g. walking) -- "
                  "some audio quality impact possible, but not severe.")
        else:
            print("Motion levels look low overall -- audio should be reasonably "
                  "clean from a motion-artifact standpoint.")
    else:
        print("No motion data captured during this recording.")
    print("=" * 60)

    print(f"\nSaving audio to '{OUTPUT_FILENAME}'...")
    with wave.open(OUTPUT_FILENAME, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(safe_audio)

    print("Done.")


if __name__ == "__main__":
    main()
