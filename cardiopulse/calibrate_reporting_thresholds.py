import os
import random
from collections import defaultdict

import librosa
import numpy as np

import model_inference
from augmentation_utils import apply_random_augmentation, load_noise_pool
from spectrogram_utils import TARGET_SR, load_fixed_length


# Thresholds are selected from validation data only.
VALIDATION_SET_DIR = "split_dataset/val"
CANDIDATE_THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]
AUGMENTED_DRAWS_PER_FILE = 5
MINIMUM_COVERAGE_PERCENT = 40.0
RANDOM_SEED = 42


def load_and_preprocess_once(filepath):
    """Load a WAV file and apply the live preprocessing exactly once."""
    y_raw, _ = librosa.load(filepath, sr=TARGET_SR, mono=True)
    raw_int16 = np.clip(y_raw * 32768.0, -32768, 32767).astype(np.int16)
    return model_inference.preprocess_chunk(raw_int16)


def classify_preprocessed(y):
    """Predict without applying reporting thresholds or filtering again."""
    return model_inference.classify_preprocessed_audio(
        y,
        apply_threshold=False,
    )


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not os.path.isdir(VALIDATION_SET_DIR):
        print(
            f"ERROR: '{VALIDATION_SET_DIR}' was not found. Expected the "
            "validation folders at split_dataset/val/<ClassName>/*.wav."
        )
        return

    print("Loading models...")
    model_inference.load_models()
    classes = model_inference._config["classes"]
    noise_pool = load_noise_pool()

    print(
        "\nRunning reporting-threshold calibration on the VALIDATION set: "
        f"1 clean pass + {AUGMENTED_DRAWS_PER_FILE} augmented draws per chunk..."
    )

    # Each item is (true class, predicted class, predicted probability).
    results = []
    unique_chunk_count = 0

    for true_class in classes:
        class_dir = os.path.join(VALIDATION_SET_DIR, true_class)
        if not os.path.isdir(class_dir):
            print(f"  WARNING: validation folder missing for '{true_class}'.")
            continue

        files = sorted(
            filename
            for filename in os.listdir(class_dir)
            if filename.lower().endswith(".wav")
        )
        unique_chunk_count += len(files)

        total_examples = len(files) * (1 + AUGMENTED_DRAWS_PER_FILE)
        print(
            f"  {true_class}: {len(files)} chunks -> "
            f"{total_examples} correlated examples"
        )

        for filename in files:
            filepath = os.path.join(class_dir, filename)

            # Preprocess once, before the clean prediction and augmentation.
            y_clean = load_and_preprocess_once(filepath)

            pred_class, confidence, _ = classify_preprocessed(y_clean)
            results.append((true_class, pred_class, confidence))

            # Training also augments an already-filtered signal. These draws
            # follow the same order and are not filtered a second time.
            for _ in range(AUGMENTED_DRAWS_PER_FILE):
                y_aug = apply_random_augmentation(
                    y_clean.copy(),
                    TARGET_SR,
                    noise_pool,
                )
                y_aug = load_fixed_length(y_aug, is_array=True)
                pred_class, confidence, _ = classify_preprocessed(y_aug)
                results.append((true_class, pred_class, confidence))

    if not results:
        print("ERROR: no validation WAV files were classified.")
        return

    print(
        f"\nClassified {len(results)} correlated examples from "
        f"{unique_chunk_count} validation chunks.\n"
    )

    by_predicted_class = defaultdict(list)
    for true_class, pred_class, confidence in results:
        by_predicted_class[pred_class].append((true_class, confidence))

    print("=" * 80)
    print(" VALIDATION REPORTING-THRESHOLD DIAGNOSTIC")
    print("=" * 80)

    recommended_thresholds = {}

    for predicted_class in classes:
        entries = by_predicted_class.get(predicted_class, [])
        prediction_count = len(entries)

        print(
            f"\n--- When the model predicts '{predicted_class}' "
            f"({prediction_count} predictions) ---"
        )

        if prediction_count == 0:
            print("  No predictions were produced for this class.")
            recommended_thresholds[predicted_class] = 0.50
            continue

        if prediction_count < 15:
            print(
                "  WARNING: this class has fewer than 15 predictions, so "
                "its threshold estimate is unstable."
            )

        print(
            f"  {'Threshold':>10}"
            f"{'Coverage':>12}"
            f"{'Precision (kept)':>18}"
            f"{'Correct/Total':>16}"
        )

        selected_threshold = CANDIDATE_THRESHOLDS[0]

        for threshold in CANDIDATE_THRESHOLDS:
            kept = [
                (true_class, confidence)
                for true_class, confidence in entries
                if confidence >= threshold
            ]

            if not kept:
                print(f"  {threshold:>10.2f}{'0%':>12}{'N/A':>18}")
                continue

            correct = sum(
                1
                for true_class, _ in kept
                if true_class == predicted_class
            )
            coverage = 100.0 * len(kept) / prediction_count
            precision = 100.0 * correct / len(kept)

            print(
                f"  {threshold:>10.2f}"
                f"{coverage:>11.1f}%"
                f"{precision:>17.1f}%"
                f"{f'{correct}/{len(kept)}':>16}"
            )

            if coverage >= MINIMUM_COVERAGE_PERCENT:
                selected_threshold = threshold

        recommended_thresholds[predicted_class] = selected_threshold

    print("\n" + "=" * 80)
    print(" VALIDATION-SELECTED STARTING THRESHOLDS")
    print("=" * 80)
    print("CONFIDENCE_THRESHOLDS = {")
    for class_name in classes:
        print(f'    "{class_name}": {recommended_thresholds[class_name]:.2f},')
    print("}")

    print(
        "\nRule used: the highest candidate threshold retaining at least "
        f"{MINIMUM_COVERAGE_PERCENT:.0f}% of predictions assigned to each class."
    )
    print(
        "The augmented draws share their underlying recording with the clean "
        "validation chunk. They are correlated robustness checks and must not "
        "be treated as independent recordings."
    )
    print(
        "Copy the selected values into CONFIDENCE_THRESHOLDS in "
        "model_inference.py. Freeze them before evaluating the test set."
    )


if __name__ == "__main__":
    main()
