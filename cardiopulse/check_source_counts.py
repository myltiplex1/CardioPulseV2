"""Run this in the same folder where you ran 1_chunk_class_specific_overlap.py.

It counts the original recordings in each class folder of Organized_Heart_Audio
and lists any recording shorter than 1.5 s. Such a recording cannot give a
three-second window that is at least half real audio, so the chunker produces
no chunks from it and it never appears in the split.
"""
import os
import librosa

INPUT_DIR = "Organized_Heart_Audio"
MIN_SECONDS = 1.5  # 50% of a 3-second window

for class_name in sorted(os.listdir(INPUT_DIR)):
    class_dir = os.path.join(INPUT_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    wavs = sorted(f for f in os.listdir(class_dir) if f.endswith(".wav"))
    short = []
    for f in wavs:
        dur = librosa.get_duration(path=os.path.join(class_dir, f))
        if dur < MIN_SECONDS:
            short.append((f, round(dur, 2)))
    print(f"{class_name}: {len(wavs)} recordings in folder, "
          f"{len(short)} shorter than {MIN_SECONDS} s, "
          f"{len(wavs) - len(short)} usable")
    for f, d in short:
        print(f"    {f}  ({d} s)")
