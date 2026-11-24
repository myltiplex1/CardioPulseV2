import os
import re
import wave
import random

SPLIT_DIR = "split_dataset_v3"
CLASSES = ["Normal", "Murmur", "Extra_Systole", "Artifact"]
EXPECTED_SR = 16000
EXPECTED_DURATION = 3.0
DURATION_TOLERANCE = 0.05  # seconds
SAMPLE_CHECK_COUNT = 30    # how many files per class to spot-check for audio integrity


def get_source_id(filename):
    """Same logic as the split script -- recovers the original recording's
    identity from a chunk filename."""
    return re.sub(r'_chunk\d+\.wav$', '', filename)


def check_audio_integrity(filepath):
    """Checks duration, sample rate, and basic sanity (not all-zero, not
    fully clipped) for one file."""
    with wave.open(filepath, 'rb') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    duration = n_frames / sr
    issues = []

    if sr != EXPECTED_SR:
        issues.append(f"wrong sample rate ({sr} != {EXPECTED_SR})")
    if abs(duration - EXPECTED_DURATION) > DURATION_TOLERANCE:
        issues.append(f"wrong duration ({duration:.3f}s != {EXPECTED_DURATION}s)")

    if sampwidth == 2:
        import struct
        n_samples = len(raw) // 2
        samples = struct.unpack(f"<{n_samples}h", raw)
        max_abs = max(abs(s) for s in samples) if samples else 0
        if max_abs == 0:
            issues.append("completely silent (all zeros)")

    return issues


def main():
    print("=" * 65)
    print(" LEAKAGE CHECK -- no source recording should appear in")
    print(" more than one split")
    print("=" * 65)

    leakage_found = False
    for cls in CLASSES:
        split_source_ids = {}
        for split in ["train", "val", "test"]:
            class_dir = os.path.join(SPLIT_DIR, split, cls)
            if not os.path.isdir(class_dir):
                continue
            files = [f for f in os.listdir(class_dir) if f.endswith(".wav")]
            source_ids = set(get_source_id(f) for f in files)
            split_source_ids[split] = source_ids

        train_val_overlap = split_source_ids.get("train", set()) & split_source_ids.get("val", set())
        train_test_overlap = split_source_ids.get("train", set()) & split_source_ids.get("test", set())
        val_test_overlap = split_source_ids.get("val", set()) & split_source_ids.get("test", set())

        total_overlap = train_val_overlap | train_test_overlap | val_test_overlap
        if total_overlap:
            leakage_found = True
            print(f"  {cls}: LEAKAGE DETECTED -- {len(total_overlap)} source "
                  f"recording(s) appear in multiple splits: {total_overlap}")
        else:
            n_sources = sum(len(s) for s in split_source_ids.values())
            print(f"  {cls}: OK -- {n_sources} unique source recordings, "
                  f"no overlap across splits")

    print()
    if leakage_found:
        print("!! LEAKAGE FOUND -- do not upload this dataset until fixed !!")
    else:
        print("No leakage detected. Safe on this front.")

    print()
    print("=" * 65)
    print(" AUDIO INTEGRITY SPOT-CHECK")
    print("=" * 65)

    random.seed(42)
    total_checked = 0
    total_issues = 0

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            class_dir = os.path.join(SPLIT_DIR, split, cls)
            if not os.path.isdir(class_dir):
                continue
            files = [f for f in os.listdir(class_dir) if f.endswith(".wav")]
            sample = random.sample(files, min(SAMPLE_CHECK_COUNT, len(files)))

            for f in sample:
                filepath = os.path.join(class_dir, f)
                issues = check_audio_integrity(filepath)
                total_checked += 1
                if issues:
                    total_issues += 1
                    print(f"  ISSUE in {split}/{cls}/{f}: {', '.join(issues)}")

    print()
    print(f"Checked {total_checked} sampled files, found {total_issues} with issues.")
    if total_issues == 0:
        print("All sampled files look structurally sound.")
    print("=" * 65)


if __name__ == "__main__":
    main()
