import os
import numpy as np
import librosa
import soundfile as sf

# --- CONFIGURATION ---
INPUT_DIR = "Organized_Heart_Audio"
OUTPUT_DIR = "chunked_heart_data_v2"
CHUNK_DURATION = 3            # seconds
OVERLAP_RATIO = 0.5           # 50% overlap between consecutive windows
TARGET_SR = 16000             # match your ESP32 hardware sample rate --
                               # IMPORTANT for later spectrogram conversion:
                               # every file must share one sample rate.
MIN_REAL_AUDIO_RATIO = 0.5     # discard a trailing window if less than this
                               # fraction of it is real (non-padded) audio


def fix_length_by_looping(chunk, target_length):
    """Pads a short chunk by looping the real audio instead of zeros,
    so the model never sees a large silent stretch."""
    if len(chunk) >= target_length:
        return chunk[:target_length]
    repeats = int(np.ceil(target_length / len(chunk)))
    looped = np.tile(chunk, repeats)
    return looped[:target_length]


def slide_chop_and_save(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"ERROR: Could not find the folder '{input_dir}'.")
        return

    os.makedirs(output_dir, exist_ok=True)

    chunk_samples = int(CHUNK_DURATION * TARGET_SR)
    hop_samples = int(chunk_samples * (1 - OVERLAP_RATIO))
    min_real_samples = int(chunk_samples * MIN_REAL_AUDIO_RATIO)

    print("==================================================")
    print(" CHUNKING (50% overlap, silence-trap fixed)")
    print("==================================================\n")
    print(f"Chunk length: {CHUNK_DURATION}s  |  Hop: {hop_samples/TARGET_SR:.2f}s "
          f"|  Target SR: {TARGET_SR}Hz\n")

    total_chunks_created = 0
    class_chunk_counts = {}
    total_discarded = 0

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith('.wav'):
                continue

            filepath = os.path.join(root, file)
            class_name = os.path.basename(root)
            class_chunk_counts.setdefault(class_name, 0)

            output_class_dir = os.path.join(output_dir, class_name)
            os.makedirs(output_class_dir, exist_ok=True)

            print(f"Processing: {file} (Class: {class_name})")

            try:
                # sr=TARGET_SR resamples every file to the same rate,
                # instead of sr=None which kept each file's native rate.
                audio, sr = librosa.load(filepath, sr=TARGET_SR)
                total_samples = len(audio)
                print(f"  -> Length: {total_samples/sr:.2f}s (resampled to {sr}Hz)")

                chunk_counter = 0
                start_idx = 0

                while start_idx < total_samples:
                    end_idx = start_idx + chunk_samples
                    chunk = audio[start_idx:end_idx]
                    real_length = len(chunk)

                    if real_length < chunk_samples:
                        if real_length < min_real_samples:
                            # Less than 50% real audio in this trailing window --
                            # discard rather than pad mostly-silence.
                            print(f"  -> Discarding trailing window "
                                  f"({real_length/sr:.2f}s real audio, below "
                                  f"{MIN_REAL_AUDIO_RATIO*100:.0f}% threshold)")
                            total_discarded += 1
                            break
                        else:
                            chunk = fix_length_by_looping(chunk, chunk_samples)
                            print(f"  -> Loop-padded trailing window "
                                  f"({real_length/sr:.2f}s real audio -> "
                                  f"{CHUNK_DURATION}s via looping)")

                    assert len(chunk) == chunk_samples, "Length mismatch error!"

                    chunk_filename = f"{file.replace('.wav', '')}_chunk{chunk_counter}.wav"
                    output_filepath = os.path.join(output_class_dir, chunk_filename)
                    sf.write(output_filepath, chunk, sr)

                    chunk_counter += 1
                    total_chunks_created += 1
                    class_chunk_counts[class_name] += 1

                    start_idx += hop_samples

                print(f"  -> SUCCESS: Saved {chunk_counter} chunks\n")

            except Exception as e:
                print(f"  -> ERROR processing {file}: {e}\n")

    print("==================================================")
    print(" CHUNKING COMPLETE")
    print("==================================================")
    print("--- FINAL CHUNK TALLY PER CLASS ---")
    for class_name, count in class_chunk_counts.items():
        print(f"{class_name.capitalize()}: {count} chunks")
    print("-----------------------------------")
    print(f" GRAND TOTAL {CHUNK_DURATION}-SECOND CHUNKS: {total_chunks_created}")
    print(f" Trailing windows discarded (too much silence): {total_discarded}")
    print(f" Output Location: ./{output_dir}/")
    print("==================================================")


if __name__ == "__main__":
    slide_chop_and_save(INPUT_DIR, OUTPUT_DIR)
