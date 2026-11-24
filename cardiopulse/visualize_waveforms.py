import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from spectrogram_utils import TARGET_SR, load_and_filter
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


def plot_waveform(ax, y, sr, title, color):
    librosa.display.waveshow(y, sr=sr, color=color, alpha=0.8, ax=ax)
    ax.set_title(title)
    limit = np.max(np.abs(y)) * 1.1 if len(y) else 1.0
    ax.set_ylim(-limit, limit)


def compare_live_augmented_vs_original():
    """Shows a raw train chunk next to what it looks like AFTER passing
    through the exact same apply_random_augmentation() the training
    generator uses -- this is what the model actually trains on."""
    print("Generating waveform: raw vs live-augmented...")
    noise_pool = load_noise_pool()
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Heart Sound (Raw Waveform): Original vs. Live Augmentation",
                 fontsize=16, fontweight='bold')

    for i, cls in enumerate(CLASSES_TO_COMPARE):
        class_dir = os.path.join(TRAIN_DIR, cls)
        file_to_load = list_wavs(class_dir)[0]

        y_orig = load_and_filter(os.path.join(class_dir, file_to_load), TARGET_SR); sr = TARGET_SR
        y_aug = apply_random_augmentation(y_orig.copy(), sr, noise_pool)

        plot_waveform(axes[i, 0], y_orig, sr, f"{cls}: Original", '#1f77b4')
        axes[i, 0].set_ylabel("Amplitude")
        plot_waveform(axes[i, 1], y_aug, sr, f"{cls}: Live-Augmented (one draw)", '#d62728')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0)
    save_and_show(fig, "waveform_original_vs_augmented")


def all_classes_from_train():
    print("Generating waveform: all classes (raw train split)...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Heart Sound (Raw Waveform) Comparison Across All Classes",
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, cls in enumerate(CLASSES):
        class_dir = os.path.join(TRAIN_DIR, cls)
        file_to_load = list_wavs(class_dir)[0]

        y = load_and_filter(os.path.join(class_dir, file_to_load), TARGET_SR); sr = TARGET_SR
        color = '#2ca02c' if cls == "Normal" else '#1f77b4'
        plot_waveform(axes[i], y, sr, f"Class: {cls}", color)
        axes[i].set_ylabel("Amplitude")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0)
    save_and_show(fig, "waveform_all_classes")


def full_length_from_raw():
    print("Generating waveform: full-length raw recordings...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Full-Length Heart Sound (Raw Waveform) Comparison",
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, cls in enumerate(CLASSES):
        class_dir = os.path.join(RAW_DATA_DIR, cls)
        file_to_load = list_wavs(class_dir)[0]

        y = load_and_filter(os.path.join(class_dir, file_to_load), TARGET_SR); sr = TARGET_SR
        color = '#2ca02c' if cls == "Normal" else '#1f77b4'
        plot_waveform(axes[i], y, sr, f"Class: {cls} (Full Length)", color)
        axes[i].set_ylabel("Amplitude")
        axes[i].set_xlabel("Time (seconds)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0)
    save_and_show(fig, "waveform_full_length")


if __name__ == "__main__":
    compare_live_augmented_vs_original()
    all_classes_from_train()
    full_length_from_raw()
