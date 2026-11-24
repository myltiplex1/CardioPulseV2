import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

from spectrogram_utils import TARGET_SR, load_and_filter
from augmentation_utils import apply_random_augmentation, load_noise_pool

# --- CONFIGURATION ---
RAW_DATA_DIR = "Organized_Heart_Audio"
TRAIN_DIR = "split_dataset/train"
CLASSES = ["Normal", "Murmur", "Extra_Systole", "Artifact"]
CLASSES_TO_COMPARE = ["Murmur", "Extra_Systole", "Artifact"]
ENVELOPE_SMOOTH_SIGMA = 100  # ~6ms smoothing window at 16kHz

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


def get_envelope(y):
    analytic_signal = hilbert(y)
    amplitude_envelope = np.abs(analytic_signal)
    return gaussian_filter1d(amplitude_envelope, sigma=ENVELOPE_SMOOTH_SIGMA)


def plot_envelope(ax, y, sr, title, color, xlabel=False):
    env = get_envelope(y)
    times = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(times, env, color=color)
    ax.fill_between(times, env, color=color, alpha=0.3)
    ax.set_title(title)
    ax.set_ylim(0, np.max(env) * 1.1 if len(env) else 1.0)
    if xlabel:
        ax.set_xlabel("Time (seconds)")


def compare_live_augmented_vs_original_envelope():
    print("Generating PCG envelope: raw vs live-augmented...")
    noise_pool = load_noise_pool()
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Heart Sound (PCG Envelope): Original vs. Live Augmentation",
                 fontsize=16, fontweight='bold')

    for i, cls in enumerate(CLASSES_TO_COMPARE):
        class_dir = os.path.join(TRAIN_DIR, cls)
        file_to_load = list_wavs(class_dir)[0]

        y_orig = load_and_filter(os.path.join(class_dir, file_to_load), TARGET_SR); sr = TARGET_SR
        y_aug = apply_random_augmentation(y_orig.copy(), sr, noise_pool)

        plot_envelope(axes[i, 0], y_orig, sr, f"{cls}: Original PCG Envelope",
                     '#1f77b4', xlabel=True)
        axes[i, 0].set_ylabel("Amplitude")
        plot_envelope(axes[i, 1], y_aug, sr, f"{cls}: Live-Augmented (one draw)",
                     '#d62728', xlabel=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0)
    save_and_show(fig, "envelope_original_vs_augmented")


def all_classes_from_train_envelope():
    print("Generating PCG envelope: all classes (raw train split)...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Heart Sound (PCG Envelope) Comparison Across All Classes",
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, cls in enumerate(CLASSES):
        class_dir = os.path.join(TRAIN_DIR, cls)
        file_to_load = list_wavs(class_dir)[0]

        y = load_and_filter(os.path.join(class_dir, file_to_load), TARGET_SR); sr = TARGET_SR
        color = '#2ca02c' if cls == "Normal" else '#1f77b4'
        plot_envelope(axes[i], y, sr, f"Class: {cls} (Envelope)", color, xlabel=True)
        axes[i].set_ylabel("Amplitude")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0)
    save_and_show(fig, "envelope_all_classes")


def full_length_from_raw_envelope():
    print("Generating PCG envelope: full-length raw recordings...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Full-Length Heart Sound (PCG Envelope) Comparison",
                 fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, cls in enumerate(CLASSES):
        class_dir = os.path.join(RAW_DATA_DIR, cls)
        file_to_load = list_wavs(class_dir)[0]

        y = load_and_filter(os.path.join(class_dir, file_to_load), TARGET_SR); sr = TARGET_SR
        color = '#2ca02c' if cls == "Normal" else '#1f77b4'
        plot_envelope(axes[i], y, sr, f"Class: {cls} (PCG Envelope)", color, xlabel=True)
        axes[i].set_ylabel("Amplitude Envelope")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=2.0)
    save_and_show(fig, "envelope_full_length")


if __name__ == "__main__":
    compare_live_augmented_vs_original_envelope()
    all_classes_from_train_envelope()
    full_length_from_raw_envelope()
