# =========================================
# LOCAL DEVELOPMENT FLASK APP (port=8000)
# Do not use this file for Render deployment.
# =========================================

import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import joblib
from utils.preprocess import extract_features

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

# === Load all models on startup ===
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ---- Diabetes model (.h5 + scaler) ----
diabetes_model_path = os.path.join(MODELS_DIR, "model_baseline_dijabetes.h5")
diabetes_model = tf.keras.models.load_model(diabetes_model_path)
scaler_path = os.path.join(MODELS_DIR, "scaler_baseline.pkl")
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

# ---- Heart models ----
heart_keras_path = os.path.join(MODELS_DIR, "model_heart_8f.h5")
heart_model = tf.keras.models.load_model(heart_keras_path)
heart_rf_path = os.path.join(MODELS_DIR, "model_heart_rf.joblib")
heart_rf = joblib.load(heart_rf_path) if os.path.exists(heart_rf_path) else None

# ---- Melanoma model ----
melanoma_model_path = os.path.join(MODELS_DIR, "efficientnet_isic2019.h5")
melanoma_model = tf.keras.models.load_model(melanoma_model_path)

# ---- Stroke (not implemented yet) ----
STROKE_FEATURES = ["Age", "Hypertension", "HeartDisease", "AvgGlucoseLevel", "BMI"]  # placeholder

# === Feature schemas ===
# ✅ Updated to match the actual Diabetes form
DIABETES_FEATURES = [
    "HighChol", "BMI", "Smoker", "HeartDiseaseorAttack",
    "PhysActivity", "GenHlth", "PhysHlth", "DiffWalk", "Age"
]

HEART_FEATURES = ["age", "sex", "cp", "thalach", "ca", "oldpeak", "thal", "slope"]


# === Helper functions ===
def _to_prob(y):
    """Ensure float probability from Keras or sklearn output"""
    if isinstance(y, (list, tuple, np.ndarray)):
        val = float(np.ravel(y)[0])
    else:
        val = float(y)
    return max(0.0, min(1.0, val))


def _extract_ordered_features(data: dict, feature_list: list):
    """Extracts feature values from dict in specific order"""
    try:
        values = []
        for feature in feature_list:
            val = data.get(feature)
            if val is None or val == "":
                raise ValueError(f"Missing feature: {feature}")
            values.append(float(val))
        return np.array(values).reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Feature extraction error: {e}")


# === ROUTES ===

@app.route("/health", methods=["GET"])
def health():
    """Quick health check route"""
    return jsonify({"status": "ok"}), 200


# ---------- DIABETES ----------
@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        # ✅ Extract and preprocess
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


# ---------- HEART ----------
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

# ---------- MELANOMA ----------
@app.route("/predict/melanoma", methods=["POST"])
def predict_melanoma():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        file = request.files["image"]
        img_bytes = io.BytesIO(file.read())

        # ✅ Resize na tačnu veličinu za EfficientNet
        img = keras_image.load_img(img_bytes, target_size=(160, 160))
        arr = keras_image.img_to_array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)

        # ✅ Predikcija
        prob = _to_prob(melanoma_model.predict(arr, verbose=0))
        label = "Malignant" if prob >= 0.5 else "Benign"

        return jsonify({
            "prediction": label,
            "confidence": round(prob, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ---------- STROKE (placeholder) ----------
@app.route("/predict/stroke", methods=["POST"])
def predict_stroke():
    try:
        data = request.get_json(force=True)
        X = _extract_ordered_features(data, STROKE_FEATURES)
        return jsonify({
            "message": "Stroke prediction model not yet implemented.",
            "received_features": X.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    import sys
    print("🧠 __name__ =", __name__)
    print("✅ Starting Flask manually (not via flask run)")

    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Flask backend running on port {port}")
    print(f"Python version: {sys.version}")

    app.run(host="127.0.0.1", port=port, debug=True, use_reloader=False)