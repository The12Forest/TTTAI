# app.py
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import json

app = Flask(__name__)

# Load model once at startup
model = tf.keras.models.load_model("models/ttt_model_perfect.h5")

@app.route("/predict", methods=["POST", "GET"])
def predict():
    # Get values from request parameters or JSON body
    values_param = request.args.get('values') or request.form.get('values')
    
    # Also try JSON body for POST requests
    if not values_param and request.is_json:
        data = request.get_json()
        if data and 'values' in data:
            values = data['values']
            if isinstance(values, list) and len(values) == 9:
                try:
                    input_array = np.array([values], dtype=np.float32)
                    output = model.predict(input_array, verbose=0)
                    return jsonify({"values": output[0].tolist()})
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
    
    if not values_param:
        return jsonify({"error": "Missing 'values' parameter"}), 400

    try:
        values_str = values_param.strip()
        
        # Try to parse as JSON array first
        if values_str.startswith('['):
            values = json.loads(values_str)
        else:
            # Parse comma-separated values
            values = [float(x.strip()) for x in values_str.split(',') if x.strip()]
                
    except (ValueError, json.JSONDecodeError) as e:
        return jsonify({
            "error": f"Could not parse values. Error: {str(e)}", 
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
