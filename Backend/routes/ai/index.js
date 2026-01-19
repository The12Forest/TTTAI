import express from "express";
import * as tf from "@tensorflow/tfjs";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const router = express.Router();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

let model = null;
let modelReady = false;

// Load the trained model on startup
async function loadModel() {
    try {
        const modelsDir = path.join(__dirname, 'models');
        const modelJsonPath = path.join(modelsDir, 'model.json');
        const weightsBinPath = path.join(modelsDir, 'group1-shard1of1.bin');
        
        console.log(`Loading AI model from: ${modelJsonPath}`);
        
        if (!fs.existsSync(modelJsonPath) || !fs.existsSync(weightsBinPath)) {
            throw new Error('Model files not found');
        }
        
        const modelJSON = JSON.parse(fs.readFileSync(modelJsonPath, 'utf8'));
        const weightsData = fs.readFileSync(weightsBinPath);
        
        model = await tf.loadLayersModel(
            tf.io.fromMemory(modelJSON.modelTopology, weightsData)
        );
        
        modelReady = true;
        console.log("✅ AI Model loaded successfully!");
        
    } catch (error) {
        console.error("❌ Failed to load AI model:", error.message);
        process.exit(1);
    }
}

loadModel();

// Health check endpoint
router.get("/status", (req, res) => {
    res.json({
        ready: modelReady,
        message: modelReady ? "AI Ready" : "AI Loading"
    });
});

// Get AI move endpoint
router.post("/move", async (req, res) => {
    try {
        if (!modelReady || !model) {
            return res.status(503).json({
                success: false,
                error: "AI Model not loaded"
            });
        }
        
        const { board } = req.body;
        
        if (!Array.isArray(board) || board.length !== 9) {
            return res.status(400).json({
                success: false,
                error: "Invalid board format"
            });
        }
        
        // Get model predictions
        const input = tf.tensor2d([board], [1, 9]);
        const predictions = model.predict(input);
        const predData = await predictions.data();
        
        // Find best available move
        const availableMoves = board
            .map((cell, idx) => cell === 0 ? idx : null)
            .filter(idx => idx !== null);
        
        if (availableMoves.length === 0) {
            input.dispose();
            predictions.dispose();
            return res.json({ success: true, move: null });
        }
        
        let bestMove = availableMoves[0];
        let bestValue = predData[availableMoves[0]];
        
        for (let move of availableMoves) {
            if (predData[move] > bestValue) {
                bestValue = predData[move];
                bestMove = move;
            }
        }
        
        input.dispose();
        predictions.dispose();
        
        res.json({
            success: true,
            move: bestMove,
            confidence: bestValue
        });
        
    } catch (error) {
        console.error("Error in /move endpoint:", error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

export { router };
