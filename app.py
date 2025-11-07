# =========================================
# PRODUCTION APP FOR RENDER DEPLOY
# =========================================

import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import joblib
from tensorflow.keras.preprocessing import image as keras_image

app = Flask(__name__)
CORS(app)

# === Paths ===
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# === Load Models ===

# ✅ Heart model (Keras format ONLY)
heart_model_path = os.path.join(MODELS_DIR, "model_heart_8f.keras")
heart_model = tf.keras.models.load_model(heart_model_path, compile=False)

# ✅ Diabetes model (.keras + scaler)
diabetes_model_path = os.path.join(MODELS_DIR, "model_baseline_dijabetes.keras")
diabetes_model = tf.keras.models.load_model(diabetes_model_path, compile=False)

scaler_path = os.path.join(MODELS_DIR, "scaler_baseline.pkl")
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None


# ✅ Melanoma model - privremeno isključen zbog TF kompatibilnosti
melanoma_model = None  

# ✅ Scaler (optional)
scaler_path = os.path.join(MODELS_DIR, "scaler_baseline.pkl")
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None


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
        vals.append(float(data[feature]))
    return np.array(vals).reshape(1, -1)


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
        # Pretvori prazne vrednosti u nulu (ili bilo koji default)
        X = [[float(v) if v != "" else 0.0 for v in row] for row in X]
        prob = _to_prob(heart_model.predict(X, verbose=0))
        label = "Positive" if prob >= 0.5 else "Negative"
        return jsonify({
            "prediction": label,
            "confidence": round(prob, 3)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# === PREDICT DIABETES (active) ===
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

# === MELANOMA DISABLED (until Keras version ready) ===
@app.route("/predict/melanoma", methods=["POST"])
def predict_melanoma():
    return jsonify({"error": "Melanoma model unavailable"}), 501


# === STROKE PLACEHOLDER ===
@app.route("/predict/stroke", methods=["POST"])
def predict_stroke():
    return jsonify({"error": "Stroke model not implemented yet"}), 501


# === MAIN ENTRY (Render) ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
