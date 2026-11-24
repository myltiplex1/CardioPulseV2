import os
import re
import numpy as np

from bpm_extraction import estimate_bpm

RAW_DATA_DIR = "Organized_Heart_Audio"
MAX_FILES_PER_GROUP = 15

SET_A_PATTERN = re.compile(r'^\w+__\d+\.wav$')  # e.g. normal__201102081321.wav


def classify_source(filename):
    return "Set A" if SET_A_PATTERN.match(filename) else "Set B"


def test_group(class_name, source_filter):
    class_dir = os.path.join(RAW_DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        return []

    all_files = sorted(f for f in os.listdir(class_dir) if f.endswith(".wav"))
    matching = [f for f in all_files if classify_source(f) == source_filter][:MAX_FILES_PER_GROUP]

    results = []
    for f in matching:
        filepath = os.path.join(class_dir, f)
        try:
            result = estimate_bpm(filepath)
            results.append({
                "file": f,
                "bpm": result["bpm_final"],
                "confidence": result["confidence"],
            })
        except Exception as e:
            print(f"    ERROR processing {f}: {e}")

    return results


def summarize(label, results):
    if not results:
        print(f"{label:<28}  no files found for this source in this class")
        return None

    confidences = [r["confidence"] for r in results]
    n = len(confidences)
    high_pct = 100 * confidences.count("high") / n
    medium_pct = 100 * confidences.count("medium") / n
    low_pct = 100 * confidences.count("low") / n

    print(f"{label:<28}n={n:<4} high={high_pct:>5.0f}%  medium={medium_pct:>5.0f}%  low={low_pct:>5.0f}%")
    return {"label": label, "n": n, "high_pct": high_pct}


def main():
    print("Controlled comparison: same recording SOURCE, different CLASS")
    print("This separates whether lower confidence is driven by the class")
    print("(rhythm irregularity) or by which part of the dataset it came from.\n")

    print("-" * 75)
    normal_a = test_group("Normal", "Set A")
    summarize("Normal (Set A)", normal_a)
    normal_b = test_group("Normal", "Set B")
    summarize("Normal (Set B)", normal_b)

    murmur_a = test_group("Murmur", "Set A")
    summarize("Murmur (Set A)", murmur_a)
    murmur_b = test_group("Murmur", "Set B")
    summarize("Murmur (Set B)", murmur_b)
    print("-" * 75)

    print("\nHow to read this:")
    print("  If Normal(Set A) >> Normal(Set B) AND Murmur(Set A) >> Murmur(Set B)")
    print("  by similar margins -- the Set A/B source is the main driver, not class.")
    print("  If Normal and Murmur look similar WITHIN each source (Set A vs Set A,")
    print("  Set B vs Set B) -- murmurs aren't meaningfully harder than Normal,")
    print("  confirming the class itself isn't the issue.")


if __name__ == "__main__":
    main()
