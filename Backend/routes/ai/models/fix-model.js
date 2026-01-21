import { readFileSync, writeFileSync } from 'fs';

const modelData = JSON.parse(readFileSync('model.json', 'utf8'));

const layers = modelData.modelTopology.model_config.config.layers;
for (let layer of layers) {
    if (layer.class_name === 'InputLayer' && layer.config.batch_shape) {
        layer.config.batchInputShape = layer.config.batch_shape;
        delete layer.config.batch_shape;
        delete layer.config.ragged;
        delete layer.config.optional;
    }

    if (layer.class_name === 'Dense') {
        // Remove regularizers entirely - they're not needed for inference
        layer.config.kernel_regularizer = null;
        layer.config.bias_regularizer = null;
        delete layer.config.quantization_config;
    }

    if (layer.class_name === 'Dropout') {
        delete layer.config.seed;
        delete layer.config.noise_shape;
    }
}

writeFileSync('model.json', JSON.stringify(modelData, null, 2));
console.log('Model fixed!');
console.log('- Fixed InputLayer');
console.log('- Removed all regularizers (not needed for inference)');
console.log('- Cleaned up unsupported fields');