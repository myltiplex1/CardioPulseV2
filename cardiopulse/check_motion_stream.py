import socket
import struct
import time
from collections import deque

# --- CONFIGURATION ---
ESP32_IP = "192.168.4.1"
TCP_PORT_MOTION = 3334
SAMPLE_RATE_HZ = 50  # matches MOTION_READ_INTERVAL_MS = 20 in firmware

# Accelerometer sensitivity: default MPU6050 range is +/-2g, so raw
# values divide by 16384 to get g's -- same conversion as the firmware
# validation sketch used.
ACCEL_SCALE = 16384.0

# Rolling window for the motion score (short window -- matches the
# motion-artifact flagging use case discussed: "is there a lot of
# movement RIGHT NOW"). Activity-context would use a much longer window
# (several minutes) over this same underlying magnitude stream --
# built separately once this core piece is confirmed working.
WINDOW_SECONDS = 2
WINDOW_SIZE = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)

# Motion score above this = "moving", below = "still". A starting
# guess, not yet tuned against real data -- expect to adjust this once
# you see real values from holding still vs. deliberately shaking it.
MOTION_THRESHOLD = 0.05


def compute_magnitude(ax_raw, ay_raw, az_raw):
    ax = ax_raw / ACCEL_SCALE
    ay = ay_raw / ACCEL_SCALE
    az = az_raw / ACCEL_SCALE
    return (ax ** 2 + ay ** 2 + az ** 2) ** 0.5


def rolling_std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance ** 0.5


def main():
    print(f"Connecting to {ESP32_IP}:{TCP_PORT_MOTION}...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ESP32_IP, TCP_PORT_MOTION))
    except Exception as e:
        print(f"Error connecting: {e}")
        print("Check the motion server is running (upload cardiopulse_firmware.ino) "
              "and both devices are on the same WiFi network.")
        return

    print("Connected. Reading motion data continuously (Ctrl+C to stop).\n")
    print("Try holding the device still, then shaking it, to see the score react.\n")

    sock.settimeout(1.0)
    magnitude_window = deque(maxlen=WINDOW_SIZE)

    try:
        while True:
            try:
                # 3 x int16 = 6 bytes per motion packet
                data = sock.recv(6)
            except socket.timeout:
                print("No data received in 1s -- check connection.")
                continue

            if len(data) < 6:
                continue

            ax_raw, ay_raw, az_raw = struct.unpack('<hhh', data)
            magnitude = compute_magnitude(ax_raw, ay_raw, az_raw)
            magnitude_window.append(magnitude)

            motion_score = rolling_std(list(magnitude_window))
            is_moving = motion_score > MOTION_THRESHOLD

            status = "MOVING" if is_moving else "still "
            print(f"\rMagnitude={magnitude:.3f}g  Motion score (std)={motion_score:.4f}  [{status}]",
                  end='', flush=True)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
