import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from spectrogram_utils import TARGET_SR, audio_to_logmel, N_MELS, FMAX, load_and_filter
from augmentation_utils import apply_random_augmentation, load_noise_pool

# --- CONFIGURATION ---
RAW_DATA_DIR = "Organized_Heart_Audio"
TRAIN_DIR = "split_dataset/train"
CLASSES = ["Normal", "Murmur", "Extra_Systole", "Artifact"]
CLASSES_TO_COMPARE = ["Murmur", "Extra_Systole", "Artifact"]

FIGURES_DIR = "report_figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_and_show(fig, name):
    """Displays the figure, then saves it as BOTH PNG and PDF."""
    png_path = os.path.join(FIGURES_DIR, f"{name}.png")
    pdf_path = os.path.join(FIGURES_DIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.show()
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def list_wavs(directory):
    return sorted(f for f in os.listdir(directory) if f.endswith(".wav"))


def plot_spectrogram(y, sr, title, ax):
    S_dB = audio_to_logmel(y, sr)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel',
                                    sr=sr, fmax=FMAX, ax=ax, cmap='magma')
    ax.set_title(title)
    return img


def scenario_1_full_length():
    print("Scenario 1: full-length raw spectrograms...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Scenario 1: Full-Length Mel-Spectrograms", fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, cls in enumerate(CLASSES):
        class_dir = os.path.join(RAW_DATA_DIR, cls)
        file_to_load = list_wavs(class_dir)[0]
        y = load_and_filter(os.path.join(class_dir, file_to_load), TARGET_SR); sr = TARGET_SR
        plot_spectrogram(y, sr, f"Class: {cls}", axes[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_and_show(fig, "spectrogram_full_length")


def scenario_2_live_augmented_vs_original():
    """Shows the exact live-augmentation pipeline the model trains on,
    as a spectrogram -- the representation the CNN actually sees."""
    print("Scenario 2: raw vs live-augmented spectrograms...")
    noise_pool = load_noise_pool()
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Scenario 2: Mel-Spectrograms (Original vs. Live Augmentation)",
                 fontsize=16, fontweight='bold')

    for i, cls in enumerate(CLASSES_TO_COMPARE):
        class_dir = os.path.join(TRAIN_DIR, cls)
        file_to_load = list_wavs(class_dir)[0]

        y_orig = load_and_filter(os.path.join(class_dir, file_to_load), TARGET_SR); sr = TARGET_SR
        y_aug = apply_random_augmentation(y_orig.copy(), sr, noise_pool)

        plot_spectrogram(y_orig, sr, f"{cls}: Original", axes[i, 0])
        plot_spectrogram(y_aug, sr, f"{cls}: Live-Augmented (one draw)", axes[i, 1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_and_show(fig, "spectrogram_original_vs_augmented")


def scenario_3_all_classes():
    print("Scenario 3: all classes (raw train split)...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Scenario 3: Mel-Spectrogram Comparison Across All Classes",
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, cls in enumerate(CLASSES):
        class_dir = os.path.join(TRAIN_DIR, cls)
        file_to_load = list_wavs(class_dir)[0]
        y = load_and_filter(os.path.join(class_dir, file_to_load), TARGET_SR); sr = TARGET_SR
        plot_spectrogram(y, sr, f"Class: {cls}", axes[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_and_show(fig, "spectrogram_all_classes")


if __name__ == "__main__":
    print("Starting Spectrogram Visualizer Dashboard...")
    scenario_1_full_length()
    scenario_2_live_augmented_vs_original()
    scenario_3_all_classes()
    print("All visualizations complete.")
