import socket
import struct
import threading
import time
from collections import deque

ACCEL_SCALE = 16384.0
MOTION_WINDOW_SECONDS = 2
MOTION_SAMPLE_RATE_HZ = 50
MOTION_WINDOW_SIZE = int(MOTION_SAMPLE_RATE_HZ * MOTION_WINDOW_SECONDS)

# 3-tier motion classification -- provisional starting point from real
# still-vs-walking data, see record_combined_audio_motion.py for details.
MOTION_TIER_LOW_MAX = 0.02
MOTION_TIER_MODERATE_MAX = 0.10


def classify_motion_tier(score):
    if score < MOTION_TIER_LOW_MAX:
        return "low"
    elif score < MOTION_TIER_MODERATE_MAX:
        return "moderate"
    else:
        return "high"


class AudioReceiver(threading.Thread):
    """Continuously reads raw int16 audio from the ESP32's audio server
    into an in-memory buffer. Runs independently in its own thread so it
    can never be slowed down by motion handling or any other processing."""

    def __init__(self, ip, port):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))
        self.sock.settimeout(0.5)
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.running = False
        self.stop_flag = threading.Event()
        self.last_data_time = None

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
                        self.last_data_time = time.time()
            except socket.timeout:
                continue
            except OSError:
                break

    def is_actively_receiving(self, max_age_sec=2.0):
        with self.lock:
            if self.last_data_time is None:
                return False
            return (time.time() - self.last_data_time) <= max_age_sec

    def get_buffer_copy(self):
        with self.lock:
            return bytes(self.buffer)

    def get_recent_samples(self, n_samples):
        """Returns the most recent n_samples int16 samples as a list,
        without clearing the buffer -- used for continuous rolling-window
        classification."""
        with self.lock:
            n_bytes = n_samples * 2
            if len(self.buffer) < n_bytes:
                return None
            recent = self.buffer[-n_bytes:]
        return struct.unpack(f'<{n_samples}h', recent)

    def clear_buffer(self):
        with self.lock:
            self.buffer = bytearray()

    def close(self):
        self.stop_flag.set()
        self.sock.close()


class MotionReceiver(threading.Thread):
    """Continuously reads raw accelerometer values from the ESP32's
    motion server, maintaining a rolling motion score. Runs independently
    from AudioReceiver -- neither can block or slow down the other.

    Stores (wall_clock_time, score) pairs so motion can be queried two
    different ways: over a SPECIFIC recent window (for motion-artifact
    flagging, tied to a particular cardiac-parameter analysis) and over
    a much LONGER window (for activity context -- resting vs. active),
    independent of when any particular recording started."""

    ACTIVITY_CONTEXT_WINDOW_SEC = 180  # 3 minutes
    ACTIVITY_THRESHOLD = 0.03          # avg score above this -> "active"

    def __init__(self, ip, port):
        super().__init__(daemon=True)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))
        self.sock.settimeout(0.5)
        self.magnitude_window = deque(maxlen=MOTION_WINDOW_SIZE)
        # Bounded by time, not count -- old entries naturally fall out
        # once they're older than the longest window we ever query.
        max_entries = int(self.ACTIVITY_CONTEXT_WINDOW_SEC * MOTION_SAMPLE_RATE_HZ * 1.2)
        self.timestamped_scores = deque(maxlen=max_entries)
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.recording_start_time = None
        self.current_score = 0.0
        self.current_tier = "low"
        self.last_data_time = None

    def mark_recording_start(self, start_time):
        self.recording_start_time = start_time

    def is_actively_receiving(self, max_age_sec=2.0):
        with self.lock:
            if self.last_data_time is None:
                return False
            return (time.time() - self.last_data_time) <= max_age_sec

    def run(self):
        while not self.stop_flag.is_set():
            try:
                data = self.sock.recv(6)
            except socket.timeout:
                continue
            except OSError:
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
                    self.current_score = score
                    self.current_tier = classify_motion_tier(score)
                    # Always store with an absolute wall-clock timestamp,
                    # regardless of whether a recording is active -- this
                    # is what lets activity context work independent of
                    # any specific recording session.
                    self.timestamped_scores.append((time.time(), score))
                self.last_data_time = time.time()

    def get_current_status(self):
        with self.lock:
            return self.current_score, self.current_tier

    def get_scores_in_window(self, start_time, end_time):
        """Motion scores between two absolute wall-clock times -- used
        for motion-artifact flagging tied to a specific cardiac-parameter
        analysis window."""
        with self.lock:
            return [s for t, s in self.timestamped_scores if start_time <= t <= end_time]

    def get_activity_context(self, window_sec=None):
        """Average motion over a long recent window, classified as
        resting or active -- context for interpreting a BPM reading
        (130 BPM means something very different at rest vs. after
        walking). Independent of any specific recording's timing."""
        window_sec = window_sec or self.ACTIVITY_CONTEXT_WINDOW_SEC
        now = time.time()
        with self.lock:
            recent = [s for t, s in self.timestamped_scores if now - t <= window_sec]

        if not recent:
            return None, "unknown"

        avg_score = sum(recent) / len(recent)
        label = "active" if avg_score > self.ACTIVITY_THRESHOLD else "resting"
        return avg_score, label

    def get_scores_during(self, duration_seconds):
        """Kept for backward compatibility with the standalone recording
        script -- scores during the current recording, relative to its
        own start time."""
        if self.recording_start_time is None:
            return []
        start = self.recording_start_time
        end = start + duration_seconds
        return self.get_scores_in_window(start, end)

    def close(self):
        self.stop_flag.set()
        self.sock.close()
