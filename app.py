# =========================================
# PRODUCTION APP FOR RENDER DEPLOY
# =========================================

import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing import image as keras_image

# === APP CONFIG ===
app = Flask(__name__)
CORS(app)

# === Disable GPU (Render nema CUDA) ===
tf.config.set_visible_devices([], 'GPU')

# === Paths ===
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# === Load Models ===

# 💓 HEART MODEL (8 feature Keras model)
heart_model_path = os.path.join(MODELS_DIR, "model_heart_8f.keras")
heart_model = tf.keras.models.load_model(heart_model_path, compile=False)
heart_model.run_eagerly = False

# 💉 DIABETES MODEL (.keras + optional scaler)
diabetes_model_path = os.path.join(MODELS_DIR, "model_baseline_dijabetes.keras")
diabetes_model = tf.keras.models.load_model(diabetes_model_path, compile=False)
diabetes_model.run_eagerly = False

scaler_path = os.path.join(MODELS_DIR, "scaler_baseline.pkl")
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# 🌞 MELANOMA MODEL (CNN Keras model + label encoder)
melanoma_model_path = os.path.join(MODELS_DIR, "skin_lesion_model.keras")
melanoma_labels_path = os.path.join(MODELS_DIR, "skin_lesion_labels.joblib")

melanoma_model = tf.keras.models.load_model(melanoma_model_path, compile=False) if os.path.exists(melanoma_model_path) else None
melanoma_labels = joblib.load(melanoma_labels_path) if os.path.exists(melanoma_labels_path) else None


# === Feature Schemas ===
DIABETES_FEATURES = [
    "HighChol", "BMI", "Smoker", "HeartDiseaseorAttack",
    "PhysActivity", "GenHlth", "PhysHlth", "DiffWalk", "Age"
]

HEART_FEATURES = ["age", "sex", "cp", "thalach", "ca", "oldpeak", "thal", "slope"]

STROKE_FEATURES = ["Age", "Hypertension", "HeartDisease", "AvgGlucoseLevel", "BMI"]


# === Helpers ===
def _to_prob(y):
    """Extract float probability."""
    return float(np.ravel(y)[0])


def _extract_ordered_features(data: dict, feature_list: list):
    """Safely creates ordered np array from request JSON."""
    vals = []
    for feature in feature_list:
        if feature not in data:
            raise ValueError(f"Missing feature: {feature}")
        vals.append(float(data[feature]) if str(data[feature]).strip() != "" else 0.0)
    return np.array([vals], dtype=float)


# === HEALTH CHECK ===
@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "Backend API running ✅"}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# === PREDICT HEART ===
@app.route("/predict/heart", methods=["POST"])
def predict_heart():
    try:
        data = request.get_json(force=True)
        X = _extract_ordered_features(data, HEART_FEATURES)

        prob = _to_prob(heart_model.predict(X, verbose=0))
        label = "Positive" if prob >= 0.5 else "Negative"

        return jsonify({
            "prediction": label,
            "confidence": round(prob, 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# === PREDICT DIABETES ===
@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.get_json(force=True)
        X = _extract_ordered_features(data, DIABETES_FEATURES)

        if scaler is not None:
            X = scaler.transform(X)

        prob = _to_prob(diabetes_model.predict(X, verbose=0))
        label = "Positive" if prob >= 0.5 else "Negative"

        return jsonify({
            "prediction": label,
            "confidence": round(prob, 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# === PREDICT MELANOMA ===
@app.route("/predict/melanoma", methods=["POST"])
def predict_melanoma():
    try:
        if melanoma_model is None or melanoma_labels is None:
            return jsonify({"error": "Melanoma model unavailable"}), 501

        if 'file' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files['file']
        img_bytes = file.read()
        img = keras_image.load_img(
            io.BytesIO(img_bytes),
            target_size=(224, 224)
        )
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        preds = melanoma_model.predict(img_array, verbose=0)
        class_idx = int(np.argmax(preds))
        label = melanoma_labels[class_idx]
        confidence = float(np.max(preds))

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# === PREDICT STROKE (placeholder) ===
@app.route("/predict/stroke", methods=["POST"])
def predict_stroke():
    return jsonify({"error": "Stroke model not implemented yet"}), 501


# === MAIN ENTRY (Render) ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)