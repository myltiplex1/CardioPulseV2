import os
import numpy as np

from bpm_extraction import estimate_bpm

# --- CONFIGURATION ---
RAW_DATA_DIR = "Organized_Heart_Audio"
CLASSES_TO_TEST = ["Normal", "Murmur", "Extra_Systole", "Artifact"]
MAX_FILES_PER_CLASS = 15  # cap for a quick, manageable run


def test_class(class_name):
    class_dir = os.path.join(RAW_DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        print(f"  WARNING: '{class_dir}' not found, skipping.")
        return []

    files = sorted(f for f in os.listdir(class_dir) if f.endswith(".wav"))[:MAX_FILES_PER_CLASS]
    results = []

    for f in files:
        filepath = os.path.join(class_dir, f)
        try:
            result = estimate_bpm(filepath)
            results.append({
                "file": f,
                "bpm": result["bpm_final"],
                "confidence": result["confidence"],
                "window_agreement": result["window_agreement_fraction"],
                "methods_agree": result["methods_agree"],
            })
        except Exception as e:
            print(f"    ERROR processing {f}: {e}")

    return results


def summarize(class_name, results):
    print(f"\n{'=' * 70}")
    print(f" {class_name}  ({len(results)} files tested)")
    print(f"{'=' * 70}")

    if not results:
        print("  No results.")
        return

    print(f"{'File':<35}{'BPM':>8}{'WinAgree':>10}{'Confidence':>12}")
    print("-" * 70)
    for r in results:
        bpm_str = f"{r['bpm']:.1f}" if r['bpm'] else "N/A"
        win_str = f"{r['window_agreement']:.2f}" if r['window_agreement'] is not None else "N/A"
        print(f"{r['file']:<35}{bpm_str:>8}{win_str:>10}{r['confidence']:>12}")

    valid_bpms = [r["bpm"] for r in results if r["bpm"] is not None]
    confidences = [r["confidence"] for r in results]

    print("-" * 70)
    if valid_bpms:
        print(f"BPM range: {min(valid_bpms):.1f} - {max(valid_bpms):.1f}   "
              f"(mean {np.mean(valid_bpms):.1f}, std {np.std(valid_bpms):.1f})")

    n = len(confidences)
    high_pct = 100 * confidences.count("high") / n if n else 0
    medium_pct = 100 * confidences.count("medium") / n if n else 0
    low_pct = 100 * confidences.count("low") / n if n else 0
    print(f"Confidence breakdown: high={high_pct:.0f}%  medium={medium_pct:.0f}%  low={low_pct:.0f}%")

    return {
        "class": class_name,
        "n": n,
        "high_pct": high_pct,
        "medium_pct": medium_pct,
        "low_pct": low_pct,
        "bpm_std": np.std(valid_bpms) if valid_bpms else None,
    }


def main():
    print("Testing BPM extraction across all four classes.")
    print("NOTE: no ground truth exists for Murmur/Extra_Systole/Artifact --")
    print("this checks plausibility and confidence behavior, not accuracy.\n")

    summaries = []
    for cls in CLASSES_TO_TEST:
        results = test_class(cls)
        summary = summarize(cls, results)
        if summary:
            summaries.append(summary)

    print(f"\n{'=' * 70}")
    print(" CROSS-CLASS COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Class':<16}{'N':>4}{'High%':>8}{'Medium%':>9}{'Low%':>7}{'BPM std':>10}")
    for s in summaries:
        std_str = f"{s['bpm_std']:.1f}" if s['bpm_std'] is not None else "N/A"
        print(f"{s['class']:<16}{s['n']:>4}{s['high_pct']:>8.0f}{s['medium_pct']:>9.0f}"
              f"{s['low_pct']:>7.0f}{std_str:>10}")

    print("\nExpected pattern if the confidence system is working correctly:")
    print("  - Murmur: similar high%/low% to Normal")
    print("  - Extra_Systole: lower high%, since rhythm irregularity genuinely")
    print("    makes a single BPM number less well-defined")
    print("  - Artifact: notably lower high% / higher low%, since these")
    print("    recordings often aren't dominated by real cardiac signal")
    print("  If Artifact instead shows mostly 'high' confidence, that's a")
    print("  real problem worth investigating -- it would mean the algorithm")
    print("  is confidently reporting numbers for non-cardiac noise.")


if __name__ == "__main__":
    main()
