import wave
import argparse
import os

def cut_wav(input_file, output_file, start_sec, end_sec):
    if not os.path.exists(input_file):
        print(f"Error: Could not find the file '{input_file}'")
        return

    with wave.open(input_file, 'rb') as infile:
        # Extract audio metadata (channels, sample width, sample rate, etc.)
        nchannels = infile.getnchannels()
        sampwidth = infile.getsampwidth()
        framerate = infile.getframerate()
        total_frames = infile.getnframes()
        total_duration = total_frames / framerate

        print(f"Original Audio: {total_duration:.2f} seconds ({framerate}Hz)")

        # Convert the time (seconds) into exactly which frames to cut
        start_frame = int(start_sec * framerate)
        end_frame = int(end_sec * framerate)

        # Safety checks
        if start_frame >= total_frames:
            print("Error: Start time is past the end of the audio file.")
            return
        if end_frame > total_frames:
            print("Notice: End time exceeds audio length. Cutting to the very end.")
            end_frame = total_frames
        if start_frame >= end_frame:
            print("Error: Start time must be less than End time.")
            return

        frames_to_read = end_frame - start_frame

        # Jump to the start time and read the exact chunk of audio
        infile.setpos(start_frame)
        audio_data = infile.readframes(frames_to_read)

    # Save the chunk into a new file
    with wave.open(output_file, 'wb') as outfile:
        outfile.setnchannels(nchannels)
        outfile.setsampwidth(sampwidth)
        outfile.setframerate(framerate)
        outfile.writeframes(audio_data)

    print(f"Success! Cut from {start_sec}s to {end_sec}s.")
    print(f"Saved as: '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut a WAV audio file based on start and end times.")
    parser.add_argument("input", help="The original WAV file (e.g., recording.wav)")
    parser.add_argument("output", help="The name for the new cut file (e.g., cut_recording.wav)")
    parser.add_argument("start", type=float, help="Start time in seconds (e.g., 5.5)")
    parser.add_argument("end", type=float, help="End time in seconds (e.g., 10.0)")

    args = parser.parse_args()
    
    cut_wav(args.input, args.output, args.start, args.end)