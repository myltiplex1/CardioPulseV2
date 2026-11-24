import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from spectrogram_utils import (
    TARGET_SR,
    EXPECTED_SAMPLES,
    despike_glitches,
    apply_bandpass,
    audio_to_logmel,
)

# Update these paths only if the deployment files are stored elsewhere.
YAMNET_MODEL_PATH = "final_deployment_yamnet.keras"
DEEP_CNN_MODEL_PATH = "final_deployment_deep_cnn.keras"
CONFIG_PATH = "final_deployment_config.json"

_yamnet_hub_model = None
_yamnet_head = None
_deep_cnn = None
_config = None


def load_models():
    """Load YAMNet, the two trained model heads, and the ensemble config."""
    global _yamnet_hub_model, _yamnet_head, _deep_cnn, _config

    for path in [YAMNET_MODEL_PATH, DEEP_CNN_MODEL_PATH, CONFIG_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"'{path}' not found. Place the saved deployment files in "
                f"this directory or update the paths at the top of this file."
            )

    print("Loading YAMNet from TF Hub (first run may take a moment)...")
    _yamnet_hub_model = hub.load("https://tfhub.dev/google/yamnet/1")

    print("Loading trained model heads...")
    _yamnet_head = tf.keras.models.load_model(YAMNET_MODEL_PATH)
    _deep_cnn = tf.keras.models.load_model(DEEP_CNN_MODEL_PATH)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        _config = json.load(f)

    print(
        f"Models loaded. Ensemble weight: {_config['yamnet_weight']} YAMNet / "
        f"{_config['cnn_weight']:.2f} CNN. Classes: {_config['classes']}"
    )


def _force_expected_length(y):
    """Return a float32 array with the exact model input length."""
    y = np.asarray(y, dtype=np.float32)

    if len(y) > EXPECTED_SAMPLES:
        y = y[:EXPECTED_SAMPLES]
    elif len(y) < EXPECTED_SAMPLES:
        y = np.pad(y, (0, EXPECTED_SAMPLES - len(y)), mode="constant")

    return y.astype(np.float32, copy=False)


def preprocess_chunk(raw_int16_samples, sr=TARGET_SR):
    """Apply the live model preprocessing exactly once.

    This function accepts raw int16 audio, converts it to floating point,
    removes isolated digital glitches, applies the 20 to 800 Hz band-pass
    filter, and fixes the input length.
    """
    y = np.asarray(raw_int16_samples, dtype=np.float32) / 32768.0
    y = despike_glitches(y)
    y = apply_bandpass(y, sr)
    return _force_expected_length(y)


# Keep these values until the corrected validation calibration has been run.
# Replace only this dictionary with the final validation-selected thresholds.
CONFIDENCE_THRESHOLDS = {
    "Normal": 0.50,
    "Murmur": 0.70,
    "Extra_Systole": 0.50,
    "Artifact": 0.70,
}


def classify_preprocessed_audio(y, apply_threshold=True):
    """Classify audio that has already been despiked and filtered.

    This entry point is used by the calibration script after it preprocesses
    each validation chunk once and creates augmented versions of that already
    filtered signal. It must not despike or band-pass filter the signal again.
    """
    if _yamnet_head is None or _deep_cnn is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")

    y = _force_expected_length(y)

    # Mean-pooled YAMNet embedding, matching training.
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)
    _, embeddings, _ = _yamnet_hub_model(waveform)
    yamnet_embedding = tf.reduce_mean(embeddings, axis=0).numpy()[np.newaxis, :]
    yamnet_probs = _yamnet_head.predict(yamnet_embedding, verbose=0)[0]

    # Deep CNN log-mel spectrogram input.
    spec = audio_to_logmel(y, TARGET_SR)[np.newaxis, ..., np.newaxis]
    cnn_probs = _deep_cnn.predict(spec, verbose=0)[0]

    yamnet_weight = _config["yamnet_weight"]
    cnn_weight = _config["cnn_weight"]
    final_probs = yamnet_weight * yamnet_probs + cnn_weight * cnn_probs

    predicted_idx = int(np.argmax(final_probs))
    predicted_class = _config["classes"][predicted_idx]
    confidence = float(final_probs[predicted_idx])

    if apply_threshold:
        class_threshold = CONFIDENCE_THRESHOLDS.get(predicted_class, 0.50)
        if confidence < class_threshold:
            predicted_class = "Uncertain"

    probabilities = dict(zip(_config["classes"], final_probs.tolist()))
    return predicted_class, confidence, probabilities


def classify_chunk(raw_int16_samples, apply_threshold=True):
    """Classify one raw audio chunk received by the live application.

    The dashboard continues to call this function. Raw audio is preprocessed
    once and then passed to the shared prediction function.
    """
    y = preprocess_chunk(raw_int16_samples)
    return classify_preprocessed_audio(y, apply_threshold=apply_threshold)
