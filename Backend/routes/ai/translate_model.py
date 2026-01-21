#!/usr/bin/env python3
"""
Script to convert TensorFlow H5 models to TensorFlow.js format
Requirements: tensorflowjs, tensorflow
Install with: pip install tensorflowjs tensorflow
"""

import os
import sys
import argparse
import tensorflowjs as tfjs
from tensorflow import keras


def convert_h5_to_tfjs(h5_path, output_dir, quantization=None):
    """
    Convert H5 model to TensorFlow.js format
    
    Args:
        h5_path: Path to the .h5 model file
        output_dir: Directory where the converted model will be saved
        quantization: Quantization type (None, 'uint8', 'uint16')
    """
    try:
        # Check if input file exists
        if not os.path.exists(h5_path):
            print(f"Error: Input file '{h5_path}' not found")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Loading model from {h5_path}...")
        model = keras.models.load_model(h5_path)
        
        print(f"Model loaded successfully")
        print(f"Converting to TensorFlow.js format...")
        
        # Convert the model
        if quantization:
            print(f"Applying {quantization} quantization...")
            tfjs.converters.save_keras_model(
                model, 
                output_dir,
                quantization_dtype_map={quantization: '*'}
            )
        else:
            tfjs.converters.save_keras_model(model, output_dir)
        
        print(f"\nConversion successful!")
        print(f"Model saved to: {output_dir}")
        print(f"\nGenerated files:")
        for file in os.listdir(output_dir):
            print(f"  - {file}")
        
        return True
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert TensorFlow H5 models to TensorFlow.js format'
    )
    parser.add_argument(
        'input',
        help='Path to the input .h5 model file'
    )
    parser.add_argument(
        'output',
        help='Output directory for the converted model'
    )
    parser.add_argument(
        '--quantization',
        choices=['uint8', 'uint16'],
        help='Quantization type for model compression (optional)'
    )
    
    args = parser.parse_args()
    
    success = convert_h5_to_tfjs(
        args.input,
        # args.output,
        "models",
        args.quantization
    )
    
    if success:
        print("\nTo load this model in Node.js:")
        print("")
        print("const tf = require('@tensorflow/tfjs-node');")
        print(f"const model = await tf.loadLayersModel('models/model.json');")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()