# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model once at startup
model = tf.keras.models.load_model("models/ttt_model_perfect.h5")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "values" not in data:
        return jsonify({"error": "Missing 'values'"}), 400

    values = data["values"]

    if not isinstance(values, list) or len(values) != 9:
        return jsonify({"error": "Input must be a list of 9 numbers"}), 400

    try:
        input_array = np.array([values], dtype=np.float32)  # (1, 9)
        output = model.predict(input_array, verbose=0)      # (1, 9)
        return jsonify({"values": output[0].tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
