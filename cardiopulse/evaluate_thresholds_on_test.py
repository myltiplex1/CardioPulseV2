import csv
import os
from collections import defaultdict

import librosa
import numpy as np

import model_inference
from spectrogram_utils import TARGET_SR


TEST_SET_DIR = "split_dataset/test"
OUTPUT_CSV = "frozen_threshold_test_results.csv"

# These values were selected using the validation set and must remain fixed.
EXPECTED_FROZEN_THRESHOLDS = {
    "Normal": 0.50,
    "Murmur": 0.70,
    "Extra_Systole": 0.50,
    "Artifact": 0.70,
}


def visible_name(class_name):
    """Use thesis-ready terminology in printed output."""
    if class_name == "Extra_Systole":
        return "Extrasystole"
    return class_name


def load_raw_int16(filepath):
    """Load one WAV file in the same format expected by live inference."""
    y, _ = librosa.load(filepath, sr=TARGET_SR, mono=True)
    return np.clip(y * 32768.0, -32768, 32767).astype(np.int16)


def calculate_raw_class_metrics(results, classes):
    """Calculate precision, recall, and F1 before reporting thresholds."""
    metrics = {}

    for class_name in classes:
        true_positive = sum(
            1
            for row in results
            if row["true_class"] == class_name
            and row["raw_prediction"] == class_name
        )
        predicted_count = sum(
            1 for row in results if row["raw_prediction"] == class_name
        )
        support = sum(
            1 for row in results if row["true_class"] == class_name
        )

        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return metrics


def print_raw_confusion_matrix(results, classes):
    print("\nRAW FOUR-CLASS CONFUSION MATRIX")
    print("Rows are true classes; columns are predicted classes.\n")

    header = "True \\ Pred".ljust(18)
    header += "".join(visible_name(c)[:12].rjust(14) for c in classes)
    print(header)

    for true_class in classes:
        row_text = visible_name(true_class)[:16].ljust(18)
        for predicted_class in classes:
            count = sum(
                1
                for row in results
                if row["true_class"] == true_class
                and row["raw_prediction"] == predicted_class
            )
            row_text += str(count).rjust(14)
        print(row_text)


def print_thresholded_confusion_matrix(results, classes):
    output_classes = list(classes) + ["Uncertain"]

    print("\nTHRESHOLDED REPORTING MATRIX")
    print("Rows are true classes; columns are displayed outputs.\n")

    header = "True \\ Output".ljust(18)
    header += "".join(visible_name(c)[:12].rjust(14) for c in output_classes)
    print(header)

    for true_class in classes:
        row_text = visible_name(true_class)[:16].ljust(18)
        for output_class in output_classes:
            count = sum(
                1
                for row in results
                if row["true_class"] == true_class
                and row["displayed_output"] == output_class
            )
            row_text += str(count).rjust(14)
        print(row_text)


def write_results_csv(results, classes):
    probability_fields = [f"probability_{class_name}" for class_name in classes]
    fieldnames = [
        "filename",
        "true_class",
        "raw_prediction",
        "confidence",
        "applied_threshold",
        "displayed_output",
        "retained",
        "raw_prediction_correct",
        "displayed_output_correct",
    ] + probability_fields

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            csv_row = {key: row[key] for key in fieldnames}
            writer.writerow(csv_row)


def main():
    if not os.path.isdir(TEST_SET_DIR):
        print(
            f"ERROR: '{TEST_SET_DIR}' was not found. Expected clean test "
            "chunks at split_dataset/test/<ClassName>/*.wav."
        )
        return

    current_thresholds = model_inference.CONFIDENCE_THRESHOLDS
    if current_thresholds != EXPECTED_FROZEN_THRESHOLDS:
        print("ERROR: model_inference.py does not contain the frozen thresholds.")
        print("Expected:")
        for class_name, threshold in EXPECTED_FROZEN_THRESHOLDS.items():
            print(f"  {class_name}: {threshold:.2f}")
        print("Current values:")
        for class_name, threshold in current_thresholds.items():
            print(f"  {class_name}: {threshold:.2f}")
        print("Update model_inference.py before running this evaluation.")
        return

    print("Frozen reporting thresholds:")
    for class_name, threshold in current_thresholds.items():
        print(f"  {visible_name(class_name)}: {threshold:.2f}")

    print("\nLoading models...")
    model_inference.load_models()
    classes = model_inference._config["classes"]

    results = []
    print("\nEvaluating the frozen thresholds on CLEAN TEST chunks only...")

    for true_class in classes:
        class_dir = os.path.join(TEST_SET_DIR, true_class)
        if not os.path.isdir(class_dir):
            print(f"  WARNING: test folder missing for '{true_class}'.")
            continue

        filenames = sorted(
            filename
            for filename in os.listdir(class_dir)
            if filename.lower().endswith(".wav")
        )
        print(f"  {visible_name(true_class)}: {len(filenames)} chunks")

        for filename in filenames:
            filepath = os.path.join(class_dir, filename)
            raw_int16 = load_raw_int16(filepath)

            raw_prediction, confidence, probabilities = (
                model_inference.classify_chunk(
                    raw_int16,
                    apply_threshold=False,
                )
            )

            applied_threshold = current_thresholds.get(raw_prediction, 0.50)
            retained = confidence >= applied_threshold
            displayed_output = raw_prediction if retained else "Uncertain"

            row = {
                "filename": filepath,
                "true_class": true_class,
                "raw_prediction": raw_prediction,
                "confidence": confidence,
                "applied_threshold": applied_threshold,
                "displayed_output": displayed_output,
                "retained": retained,
                "raw_prediction_correct": raw_prediction == true_class,
                "displayed_output_correct": displayed_output == true_class,
            }

            for class_name in classes:
                row[f"probability_{class_name}"] = probabilities[class_name]

            results.append(row)

    if not results:
        print("ERROR: no test WAV files were evaluated.")
        return

    total = len(results)
    retained_results = [row for row in results if row["retained"]]
    retained_count = len(retained_results)
    uncertain_count = total - retained_count

    raw_correct = sum(row["raw_prediction_correct"] for row in results)
    retained_correct = sum(
        row["displayed_output_correct"] for row in retained_results
    )

    raw_accuracy = raw_correct / total
    overall_coverage = retained_count / total
    retained_accuracy = (
        retained_correct / retained_count if retained_count else 0.0
    )

    print("\n" + "=" * 80)
    print(" FROZEN-THRESHOLD TEST RESULTS")
    print("=" * 80)
    print(f"Unique clean test chunks: {total}")
    print(f"Raw four-class accuracy: {raw_correct}/{total} = {raw_accuracy:.1%}")
    print(
        f"Outputs retained after thresholds: {retained_count}/{total} "
        f"= {overall_coverage:.1%}"
    )
    print(f"Outputs marked Uncertain: {uncertain_count}/{total} = {uncertain_count / total:.1%}")
    print(
        f"Accuracy among retained outputs: {retained_correct}/{retained_count} "
        f"= {retained_accuracy:.1%}"
        if retained_count
        else "Accuracy among retained outputs: N/A"
    )

    raw_metrics = calculate_raw_class_metrics(results, classes)
    print("\nRAW FOUR-CLASS CLASSIFICATION REPORT")
    print(
        f"  {'Class':<16}{'Precision':>12}{'Recall':>10}"
        f"{'F1':>10}{'Support':>10}"
    )
    for class_name in classes:
        metric = raw_metrics[class_name]
        print(
            f"  {visible_name(class_name):<16}"
            f"{metric['precision']:>12.3f}"
            f"{metric['recall']:>10.3f}"
            f"{metric['f1']:>10.3f}"
            f"{metric['support']:>10}"
        )

    print("\nTHRESHOLD PERFORMANCE BY RAW PREDICTED CLASS")
    print(
        f"  {'Class':<16}{'Predicted':>11}{'Retained':>11}"
        f"{'Coverage':>11}{'Kept precision':>17}"
    )

    predictions_by_class = defaultdict(list)
    for row in results:
        predictions_by_class[row["raw_prediction"]].append(row)

    for class_name in classes:
        class_predictions = predictions_by_class[class_name]
        predicted_count = len(class_predictions)
        kept = [row for row in class_predictions if row["retained"]]
        kept_correct = sum(row["raw_prediction_correct"] for row in kept)
        coverage = len(kept) / predicted_count if predicted_count else 0.0
        precision = kept_correct / len(kept) if kept else 0.0

        print(
            f"  {visible_name(class_name):<16}"
            f"{predicted_count:>11}"
            f"{len(kept):>11}"
            f"{coverage:>10.1%}"
            f"{precision:>16.1%}"
        )

    print("\nCOVERAGE BY TRUE CLASS")
    print(
        f"  {'Class':<16}{'Support':>10}{'Retained':>11}"
        f"{'Uncertain':>11}{'Coverage':>11}{'Retained accuracy':>20}"
    )

    for class_name in classes:
        true_rows = [row for row in results if row["true_class"] == class_name]
        kept = [row for row in true_rows if row["retained"]]
        correct = sum(row["displayed_output_correct"] for row in kept)
        coverage = len(kept) / len(true_rows) if true_rows else 0.0
        accuracy = correct / len(kept) if kept else 0.0

        print(
            f"  {visible_name(class_name):<16}"
            f"{len(true_rows):>10}"
            f"{len(kept):>11}"
            f"{len(true_rows) - len(kept):>11}"
            f"{coverage:>10.1%}"
            f"{accuracy:>19.1%}"
        )

    print_raw_confusion_matrix(results, classes)
    print_thresholded_confusion_matrix(results, classes)

    write_results_csv(results, classes)
    print(f"\nSaved individual results to: {OUTPUT_CSV}")
    print("These test results evaluate the frozen thresholds.")
    print("Do not change the thresholds after reviewing this output.")


if __name__ == "__main__":
    main()
