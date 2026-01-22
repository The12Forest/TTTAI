import * as tf from '@tensorflow/tfjs'
import express from "express";
import '@tensorflow/tfjs-backend-cpu'; // Explicitly use CPU backend
const router = express.Router();
const logprefix = "AIRouter:        ";
let model = null;

// try {
//     if (!model) {
//         // Load via HTTP URL (adjust port as needed)
//         model = await tf.loadLayersModel('http://localhost:80/api/ai/models/model.json');
//         console.log(logprefix + 'Model loaded successfully!');
//     }
// } catch (error) {
//     console.error(logprefix + 'Error loading model:', error.message);
// }


// async function loadModel() {
//     try {
//         if (!model) {
//             const modelUrl = `http://localhost/api/ai/models/model.json`;
//             console.log(logprefix + 'Loading model from:', modelUrl);
//             model = await tf.loadLayersModel(modelUrl);
//             console.log(logprefix + 'Model loaded successfully!');
//             model.summary();
//         }
//         return model;
//     } catch (error) {
//         console.error(logprefix + 'Error loading model:', error.message);
//         throw error;
//     }
// }

async function loadModel() {
    try {
        if (!model) {
            const modelUrl = `http://localhost/api/ai/models/model.json`;

            // Ensure TensorFlow.js is ready
            await tf.ready();
            console.log(logprefix + 'TensorFlow.js backend:', tf.getBackend());

            // Try loading with fetch options
            model = await tf.loadLayersModel(modelUrl, {
                strict: false // Less strict loading
            });

            console.log(logprefix + 'Model loaded successfully!');
            model.summary();
        }
        return model;
    } catch (error) {
        console.error(logprefix + 'Error loading model:', error);
        console.error(logprefix + 'Full error stack:', error.stack);
        throw error;
    }
}

// Endpoint to load the model
router.get('/load', async (req, res) => {
    try {
        // const protocol = req.protocol;
        // const host = req.get('host');
        // const baseUrl = `${protocol}://${host}`;

        await loadModel();
        res.json({ success: true, message: 'Model loaded successfully' });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

router.get("/getAIMove", async (req, res) => {
    try {
        // Load model if not already loaded
        if (!model) {
            await loadModel();
        }

        let board = JSON.parse(req.query.Board);
        let available_moves = [];

        for (let i = 0; i < 9; i++) {
            if (board[i] === 0) {
                available_moves.push(i);
            }
        }

        // Get predictions from model
        const inputTensor = tf.tensor2d([board]);
        const predictionTensor = model.predict(inputTensor);
        const predictions = await predictionTensor.array();

        // Clean up tensors
        inputTensor.dispose();
        predictionTensor.dispose();

        // Get probabilities for available moves
        let move_probs = [];
        for (let move of available_moves) {
            move_probs.push(predictions[0][move]); // Fixed: predictions[0][move]
        }

        // Find the index of the highest probability
        const chosen_index = move_probs.indexOf(Math.max(...move_probs));
        const chosen_move = available_moves[chosen_index];

        board[chosen_move] = 1;

        res.json({ "Okay": true, "Board": board }); // Fixed typo: "Board"
    } catch (error) {
        console.error(logprefix + 'getAIMove error:', error.message);
        res.status(500).json({ success: false, error: error.message });
    }
});


export { router };
/*

Create a file tensorflow.continue.py int the Gen data Folder


it shoud ask for a model path at start and it acepts a h5 model


afterwards let a human play against the AI in the Terminal. If the Human wone save the winner s play as 1 is human and -1 is AI then save eatch game move as an individual dataset as [
    [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]],
    [[Ausgang][The Position the Human Played]]
]
one time the human begins one time the ai begins
ask after eatch game if it shoud now finetune the model on the gatherd data but only ask after 100 sampels 


ten finetune the AI on the gameplay of the Human after ask if the program shoud be stoped or if it shoud repeat the whole sicle 

the ai palys with an array of the gamefield and then gives an array of 9 with the probabilety of the position the ai wants to play 