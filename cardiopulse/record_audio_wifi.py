import socket
import wave
import time

# --- CONFIGURATION ---
# ESP32_IP = "10.78.78.208"  # printed by the ESP32 over USB-serial at boot
ESP32_IP = "192.168.4.1"
TCP_PORT = 3333
SAMPLE_RATE = 16000
OUTPUT_FILENAME = "recording_1.wav"
MAX_RECORD_SECONDS = 60

print(f"Connecting to {ESP32_IP}:{TCP_PORT}...")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ESP32_IP, TCP_PORT))
except Exception as e:
    print(f"Error connecting: {e}")
    print("Check the ESP32's IP address (printed over USB-serial at boot) "
          "and that both devices are on the same WiFi network.")
    exit()

print("\n--- READY ---")
input(f"Press ENTER to start recording '{OUTPUT_FILENAME}' "
      f"(max {MAX_RECORD_SECONDS}s)...")

sock.sendall(b'r')
start_time = time.time()
print("\n[ RECORDING IN PROGRESS... ]")
print("Press Ctrl+C to stop early and save.")

raw_audio_buffer = bytearray()
sock.settimeout(0.5)

try:
    while True:
        if time.time() - start_time > MAX_RECORD_SECONDS:
            print("\n[ Max duration reached, stopping automatically ]")
            break
        try:
            chunk = sock.recv(4096)
            if chunk:
                raw_audio_buffer.extend(chunk)
        except socket.timeout:
            continue
except KeyboardInterrupt:
    print("\n[ STOPPING... ]")

elapsed_wall_clock = time.time() - start_time
sock.sendall(b's')
time.sleep(0.3)
sock.close()

safe_length = len(raw_audio_buffer) - (len(raw_audio_buffer) % 2)
safe_raw_buffer = raw_audio_buffer[:safe_length]

actual_samples = len(safe_raw_buffer) // 2
expected_samples = int(elapsed_wall_clock * SAMPLE_RATE)
dropout_pct = 100 * (1 - actual_samples / expected_samples) if expected_samples > 0 else 0

print("\n" + "=" * 55)
print(" WIFI TRANSPORT DIAGNOSTIC")
print("=" * 55)
print(f"Wall-clock recording duration: {elapsed_wall_clock:.2f}s")
print(f"Expected samples (duration x {SAMPLE_RATE}Hz): {expected_samples}")
print(f"Actual samples received:                     {actual_samples}")
print(f"Estimated dropout: {dropout_pct:+.2f}%")
if dropout_pct > 2:
    print("WARNING: notable sample loss detected -- WiFi couldn't keep up "
          "with the audio stream at some point. Check signal strength, "
          "distance, and network congestion.")
else:
    print("Dropout within normal/negligible range.")
print("=" * 55)

print(f"\nSaving to '{OUTPUT_FILENAME}'...")
with wave.open(OUTPUT_FILENAME, 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(safe_raw_buffer)

print(f"Done! Saved as '{OUTPUT_FILENAME}'.")
