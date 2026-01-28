import express from "express";
import fs from "fs";
import { Ollama } from "ollama";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

const router = express.Router();
const logprefix = "AIRouter:        ";
const conf = JSON.parse(fs.readFileSync("./config.json"));
let gameplayhistory = []


router.use("/save", (req, res) => {
    fs.writeFileSync("./Backend/saves/games_ai_history.json", JSON.stringify(gameplayhistory));
    console.log(logprefix + "History saved:     " + JSON.stringify(gameplayhistory));
    res.json({ Okay: true, Message: "History saved!" });
});

router.use("/load", (req, res) => {
    gameplayhistory = JSON.parse(fs.readFileSync("./Backend/saves/games_ai_history.json"));
    console.log(logprefix + "History loaded:    " + JSON.stringify(gameplayhistory))
    res.json({ Okay: true, Message: "History loaded!" });
});

router.get("/getAIMove", async (req, res) => {
    try {
        let board = JSON.parse(req.query.Board);
        let available_moves = [];

        for (let i = 0; i < 9; i++) {
            if (board[i] === 0) {
                available_moves.push(i);
            }
        }


        const response = await fetch('http://tttai-ai:8000/predict?values=' + JSON.stringify(board));

        const data = await response.json();
        const predictions = data.values;

        console.log(logprefix + "Asking AI: " + JSON.stringify(board))
        console.log(logprefix + "AI Answer: " + JSON.stringify(predictions))

        let move_probs = [];
        for (let move of available_moves) {
            move_probs.push(predictions[move]);
        }

        const chosen_index = move_probs.indexOf(Math.max(...move_probs));
        const chosen_move = available_moves[chosen_index];

        board[chosen_move] = -1;  

        res.json({ "Okay": true, "Board": board });
    } catch (error) {
        console.error(logprefix + 'getAIMove error:', error.message);
        res.status(500).json({ success: false, error: error.message });
    }
});

router.post("/gameplayhistory", async (req, res) => {
    const { gameplay } = req.body;
    if (gameplay && Array.isArray(gameplay)) {
        gameplayhistory.push(gameplay);
        console.log(logprefix + "Gameplay history received: " + JSON.stringify(gameplay));
        res.json({ Okay: true, Message: "History recorded!" });
    } else {
        res.status(400).json({ Okay: false, Message: "Invalid gameplay data" });
    }
})


// Not Tested
router.post("/convertList", async (req, res) => {
    try {
        if (!gameplayhistory || gameplayhistory.length === 0) {
            return res.status(400).json({ Okay: false, Message: "No gameplay history available" });
        }

        // Initialize Ollama with endpoint from config
        const ollama = new Ollama({ host: conf.OllamaEndPoint });

        // Schema for validation
        const CorrectedMoves = z.object({
            corrections: z.array(z.object({
                boardState: z.array(z.number()).length(9),
                correctPosition: z.number().min(0).max(8),
                explanation: z.string(),
            })),
        });

        const trainingExamples = [];

        // Process each game in gameplay history
        for (const game of gameplayhistory) {
            if (!Array.isArray(game) || game.length === 0) continue;

            // Send the entire game to Ollama for analysis
            const prompt = `You are analyzing a tic-tac-toe game where the AI (playing as O) lost to the human (playing as X).

Board positions layout:
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8

Values: -1 = AI (O), 1 = Human (X), 0 = empty

Here is the complete game sequence (array of board states after each move):
${JSON.stringify(game, null, 2)}

The human won this game. Analyze ALL moves and identify ONLY the critical moves where the AI played wrong and should have played differently to avoid losing.

For each wrong AI move, provide:
- boardState: the board state BEFORE the AI made the wrong move (array of 9 numbers: -1, 0, or 1)
- correctPosition: the position (0-8) where the AI SHOULD have played
- explanation: why this move was critical

Only include moves that actually matter - where a different AI move could have prevented the loss or led to a win/draw.

Return ONLY valid JSON in this exact format:
{"corrections": [{"boardState": [0,0,0,0,0,0,0,0,0], "correctPosition": 4, "explanation": "reason"}]}`;

            const response = await ollama.generate({
                model: conf.OllamaModel,
                prompt: prompt,
                format: "json",
                stream: false,
            });

            try {
                const parsed = JSON.parse(response.response);
                const result = CorrectedMoves.parse(parsed);
                
                for (const correction of result.corrections) {
                    // Create output array: 9 values (0-1), 1 at correct position, 0 elsewhere
                    const output = [0, 0, 0, 0, 0, 0, 0, 0, 0];
                    output[correction.correctPosition] = 1;
                    
                    // Format: [currentBoard, outputProbabilities]
                    trainingExamples.push([correction.boardState, output]);
                    console.log(logprefix + `Correction: position ${correction.correctPosition} - ${correction.explanation}`);
                }
            } catch (parseError) {
                console.log(logprefix + `Could not parse Ollama response: ${response.response}`);
            }
        }

        res.json({
            Okay: true,
            Message: `Generated ${trainingExamples.length} corrected moves from ${gameplayhistory.length} games`,
            examples: trainingExamples,
        });
    } catch (error) {
        console.error(logprefix + "convertList error:", error.message);
        res.status(500).json({ Okay: false, Message: error.message });
    }
});

export { router };
