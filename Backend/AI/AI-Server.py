# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model once at startup
model = tf.keras.models.load_model("models/ttt_model_perfect.h5")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    # Get values from request parameters
    values_param = request.args.get('values') or request.form.get('values')
    
    if not values_param:
        return jsonify({"error": "Missing 'values' parameter"}), 400

    try:
        # Handle array format like [0, 0, 0, 0, 0, 0, 0, 0, 0] or comma-separated
        values_str = values_param.strip()
        if values_str.startswith('[') and values_str.endswith(']'):
            values_str = values_str[1:-1]  # Remove brackets
        
        # Parse the comma-separated string into a list of numbers
        values = [float(x.strip()) for x in values_str.split(',')]
    except (ValueError, AttributeError):
        return jsonify({"error": "Values must be comma-separated numbers or array format"}), 400

    if len(values) != 9:
        return jsonify({"error": "Input must contain exactly 9 numbers"}), 400

    try:
        input_array = np.array([values], dtype=np.float32)  # (1, 9)
        output = model.predict(input_array, verbose=0)      # (1, 9)
        return jsonify({"values": output[0].tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
