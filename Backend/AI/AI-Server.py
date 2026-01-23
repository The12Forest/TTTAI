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
        
        # Remove brackets if present (handle both [ ] and URL encoded %5B %5D)
        if values_str.startswith('[') or values_str.startswith('%5B'):
            values_str = values_str.lstrip('[').lstrip('%5B')
        if values_str.endswith(']') or values_str.endswith('%5D'):
            values_str = values_str.rstrip(']').rstrip('%5D')
        
        # Split by comma and clean up each value
        values = []
        for x in values_str.split(','):
            cleaned = x.strip()
            if cleaned:  # Skip empty strings
                values.append(float(cleaned))
                
    except (ValueError, AttributeError) as e:
        return jsonify({
            "error": f"Values must be comma-separated numbers or array format. Error: {str(e)}", 
            "received": values_param
        }), 400

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
