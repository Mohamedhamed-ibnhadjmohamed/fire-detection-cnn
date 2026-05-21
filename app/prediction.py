"""
prediction.py — Fonctions de chargement du modèle et de prédiction.
"""

import numpy as np
from PIL import Image
import tensorflow as tf
import os

IMG_SIZE = 128

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Redimensionne et normalise une image PIL pour l'inférence."""
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 128, 128, 3)


def load_model(model_path: str):
    """Charge un modèle Keras sauvegardé (.keras ou .h5)."""
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)


def predict(model, image: Image.Image) -> dict:
    """
    Retourne un dict avec :
      - label      : 'Feu' ou 'Non-feu'
      - confidence : probabilité en % de la classe prédite
      - fire_prob  : probabilité brute de feu (0-1)
    """
    img_array = preprocess_image(image)
    fire_prob = float(model.predict(img_array, verbose=0)[0][0])
    label = "Feu 🔥" if fire_prob >= 0.5 else "Non-feu ✅"
    confidence = fire_prob if fire_prob >= 0.5 else 1 - fire_prob
    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "fire_prob": round(fire_prob * 100, 2),
        "safe_prob": round((1 - fire_prob) * 100, 2),
    }


def make_gradcam(model, image: Image.Image, last_conv_name: str = None):
    """
    Calcule la carte Grad-CAM pour une image.
    Retourne (heatmap_array, overlay_array) ou (None, None) si non disponible.
    """
    try:
        # Trouver automatiquement la dernière Conv2D si non spécifiée
        if last_conv_name is None:
            conv_layers = [l.name for l in model.layers
                           if isinstance(l, tf.keras.layers.Conv2D)]
            if not conv_layers:
                return None, None
            last_conv_name = conv_layers[-1]

        img_array = preprocess_image(image)

        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            img_tensor = tf.cast(img_array, tf.float32)
            conv_out, pred = grad_model(img_tensor)
            pred_score = pred[:, 0]

        grads   = tape.gradient(pred_score, conv_out)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam     = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1)
        cam     = tf.nn.relu(cam)
        cam     = cam / (tf.reduce_max(cam) + 1e-8)
        cam_np  = cam.numpy()

        # Redimensionner au format image
        import matplotlib.pyplot as plt
        cam_img = Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
            (IMG_SIZE, IMG_SIZE))
        cam_arr = np.array(cam_img) / 255.0

        # Heatmap colorée
        heatmap = plt.cm.jet(cam_arr)[:, :, :3]

        # Superposition
        original = np.array(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)),
                             dtype=np.float32) / 255.0
        overlay = np.clip(original * 0.6 + heatmap * 0.4, 0, 1)

        return (heatmap * 255).astype(np.uint8), (overlay * 255).astype(np.uint8)

    except Exception:
        return None, None
