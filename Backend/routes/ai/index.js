// import * as tf from '@tensorflow/tfjs'
import express from "express";
// import '@tensorflow/tfjs-backend-cpu'; // Explicitly use CPU backend
// const fetch = require("node-fetch");
const router = express.Router();
const logprefix = "AIRouter:        ";
let model = null;

// async function loadModel() {
//     try {
//         if (!model) {
//             const modelUrl = `http://localhost/api/ai/models/model.json`;

//             // Ensure TensorFlow.js is ready
//             await tf.ready();
//             console.log(logprefix + 'TensorFlow.js backend:', tf.getBackend());

//             // Try loading with fetch options
//             model = await tf.loadLayersModel(modelUrl, {
//                 strict: false // Less strict loading
//             });

//             console.log(logprefix + 'Model loaded successfully!');
//             model.summary();
//         }
//         return model;
//     } catch (error) {
//         console.error(logprefix + 'Error loading model:', error);
//         console.error(logprefix + 'Full error stack:', error.stack);
//         throw error;
//     }
// }

// // Endpoint to load the model
// router.get('/load', async (req, res) => {
//     try {
//         // const protocol = req.protocol;
//         // const host = req.get('host');
//         // const baseUrl = `${protocol}://${host}`;

//         await loadModel();
//         res.json({ success: true, message: 'Model loaded successfully' });
//     } catch (error) {
//         res.status(500).json({ success: false, error: error.message });
//     }
// });

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
        // const inputTensor = tf.tensor2d([board]);
        // const predictionTensor = model.predict(inputTensor);        

        const response = await fetch("http://72.62.112.40:8000/predict?values=" + JSON.stringify(board));

        const data = await response.json();
        const predictions = data.values

        // const data = await response.json();
        // predictions = await data.values

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
