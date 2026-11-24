import serial
import wave
import time

# --- CONFIGURATION ---
COM_PORT = 'COM7'          # Update if your port changes
BAUD_RATE = 921600         # MUST match Serial.begin() value in the .ino file
SAMPLE_RATE = 16000
OUTPUT_FILENAME = 'noise_clip_hs.wav'   # change per clip, e.g. noise_room_hum.wav
MAX_RECORD_SECONDS = 60    # safety cap in case you forget to press Ctrl+C

print(f"Connecting to {COM_PORT} at {BAUD_RATE} baud...")
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
except Exception as e:
    print(f"Error opening serial port: {e}")
    print("Please check if the COM port is correct and not busy.")
    exit()

time.sleep(2)  # Wait for ESP32 to reset

print("\n--- READY ---")
input(f"Press ENTER to start recording '{OUTPUT_FILENAME}' "
      f"(max {MAX_RECORD_SECONDS}s)...")

ser.reset_input_buffer()
ser.write(b'r')
print("\n[ RECORDING IN PROGRESS... ]")
print("Press Ctrl+C to stop early and save.")

raw_audio_buffer = bytearray()
start_time = time.time()

try:
    while True:
        if time.time() - start_time > MAX_RECORD_SECONDS:
            print("\n[ Max duration reached, stopping automatically ]")
            break
        waiting = ser.in_waiting
        if waiting > 0:
            raw_audio_buffer.extend(ser.read(waiting))
except KeyboardInterrupt:
    print("\n[ STOPPING... ]")

ser.write(b's')
time.sleep(0.2)

# Ensure even byte count for 16-bit PCM audio
safe_length = len(raw_audio_buffer) - (len(raw_audio_buffer) % 2)
safe_raw_buffer = raw_audio_buffer[:safe_length]

duration_sec = len(safe_raw_buffer) / 2 / SAMPLE_RATE
print(f"Caught {len(safe_raw_buffer)} raw bytes (~{duration_sec:.1f}s) of data.")

print(f"Saving to '{OUTPUT_FILENAME}'...")
with wave.open(OUTPUT_FILENAME, 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit audio
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(safe_raw_buffer)

print(f"\nDone! Saved as '{OUTPUT_FILENAME}'.")
ser.close()
