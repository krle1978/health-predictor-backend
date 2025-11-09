# =========================================
# 🧠 HEALTH PREDICTOR BACKEND – FINAL VERSION
# =========================================
# Supports: Heart, Diabetes, Stroke, Melanoma
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

# === Disable GPU (Render uses CPU only) ===
tf.config.set_visible_devices([], 'GPU')

# === Paths ===
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# === SAFE LOAD HELPERS ===
def safe_load_model(path, name):
    """Safely load a model if it exists."""
    if not os.path.exists(path):
        print(f"⚠️  {name} model not found: {os.path.basename(path)}")
        return None
    try:
        model = tf.keras.models.load_model(path, compile=False)
        model.run_eagerly = False
        print(f"✅ Loaded {name} model ({os.path.basename(path)})")
        return model
    except Exception as e:
        print(f"❌ Failed to load {name} model -> {e}")
        return None


# === LOAD MODELS ===
heart_model = safe_load_model(os.path.join(MODELS_DIR, "model_heart_8f.keras"), "Heart")
diabetes_model = safe_load_model(os.path.join(MODELS_DIR, "model_baseline_dijabetes.keras"), "Diabetes")
stroke_model = safe_load_model(os.path.join(MODELS_DIR, "model_stroke_v2.keras"), "Stroke")
melanoma_model = safe_load_model(os.path.join(MODELS_DIR, "skin_lesion_model.keras"), "Melanoma")

# === Load scaler and labels ===
scaler = None
scaler_path = os.path.join(MODELS_DIR, "scaler_baseline.pkl")
if os.path.exists(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        print("✅ Scaler loaded successfully")
    except Exception as e:
        print(f"⚠️ Failed to load scaler: {e}")

melanoma_labels = None
labels_path = os.path.join(MODELS_DIR, "skin_lesion_labels.joblib")
if os.path.exists(labels_path):
    try:
        melanoma_labels = joblib.load(labels_path)
        print(f"🏷️ Loaded {len(melanoma_labels)} melanoma labels")
    except Exception as e:
        print(f"⚠️ Failed to load melanoma labels: {e}")


# === FEATURE SCHEMAS ===
DIABETES_FEATURES = [
    "HighChol", "BMI", "Smoker", "HeartDiseaseorAttack",
    "PhysActivity", "GenHlth", "PhysHlth", "DiffWalk", "Age"
]

HEART_FEATURES = ["age", "sex", "cp", "thalach", "ca", "oldpeak", "thal", "slope"]

STROKE_FEATURES = [
    "Age", "Hypertension", "HeartDisease",
    "AvgGlucoseLevel", "BMI", "SmokingStatus",
    "WorkType", "ResidenceType"
]


# === HELPERS ===
def _to_prob(y):
    """Convert model output to single float probability."""
    return float(np.ravel(y)[0])


def _extract_ordered_features(data: dict, feature_list: list):
    """Extracts and orders features safely from JSON body."""
    vals = []
    for feature in feature_list:
        val = str(data.get(feature, "")).strip()
        vals.append(float(val) if val != "" else 0.0)
    return np.array([vals], dtype=float)


# === ROUTES ===

@app.route("/", methods=["GET"])
def root():
    """Root endpoint showing which models are loaded."""
    return jsonify({
        "status": "✅ Backend API running",
        "available_models": {
            "heart": heart_model is not None,
            "diabetes": diabetes_model is not None,
            "stroke": stroke_model is not None,
            "melanoma": melanoma_model is not None
        }
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# === HEART PREDICTION ===
@app.route("/predict/heart", methods=["POST"])
def predict_heart():
    if heart_model is None:
        return jsonify({"error": "Heart model not available"}), 501
    try:
        data = request.get_json(force=True)
        X = _extract_ordered_features(data, HEART_FEATURES)
        prob = _to_prob(heart_model.predict(X, verbose=0))
        label = "Positive" if prob >= 0.5 else "Negative"
        return jsonify({"prediction": label, "confidence": round(prob, 3)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# === DIABETES PREDICTION ===
@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    if diabetes_model is None:
        return jsonify({"error": "Diabetes model not available"}), 501
    try:
        data = request.get_json(force=True)
        X = _extract_ordered_features(data, DIABETES_FEATURES)
        if scaler is not None:
            X = scaler.transform(X)
        prob = _to_prob(diabetes_model.predict(X, verbose=0))
        label = "Positive" if prob >= 0.5 else "Negative"
        return jsonify({"prediction": label, "confidence": round(prob, 3)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# === STROKE PREDICTION ===
@app.route("/predict/stroke", methods=["POST"])
def predict_stroke():
    if stroke_model is None:
        return jsonify({"error": "Stroke model not available"}), 501
    try:
        data = request.get_json(force=True)
        X = _extract_ordered_features(data, STROKE_FEATURES)
        prob = _to_prob(stroke_model.predict(X, verbose=0))
        label = "Positive" if prob >= 0.5 else "Negative"
        return jsonify({"prediction": label, "confidence": round(prob, 3)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# === MELANOMA PREDICTION ===
@app.route("/predict/melanoma", methods=["POST"])
def predict_melanoma():
    if melanoma_model is None:
        return jsonify({"error": "Melanoma model not available"}), 501
    try:
        if "file" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["file"]
        img = keras_image.load_img(file, target_size=(160, 160))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = melanoma_model.predict(img_array, verbose=0)[0]
        top_idx = np.argmax(predictions)
        label = melanoma_labels[top_idx] if melanoma_labels else str(top_idx)
        confidence = float(predictions[top_idx])

        return jsonify({"prediction": label, "confidence": round(confidence, 3)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# === MAIN ENTRY ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running on port {port}")
    app.run(host="0.0.0.0", port=port)
