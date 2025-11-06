import tensorflow as tf
import os

model_path = os.path.join("models", "model_heart_8f.keras")

if not os.path.exists(model_path):
    print("❌ Model not found at:", model_path)
else:
    print("✅ Loading model from:", model_path)
    model = tf.keras.models.load_model(model_path)
    print("Model input shape:", model.input_shape)
