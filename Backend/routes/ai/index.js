import express from "express";
import fs from "fs";
const router = express.Router();
const logprefix = "AIRouter:        ";
let gameplayhistory = []

router.use("/save", (req, res) => {
    fs.writeFileSync("./Backend/saves/user/games_ai_history.json", JSON.stringify(gameplayhistory));
    console.log(logprefix + "History saved:     " + JSON.stringify(gameplayhistory));
    res.json({ Okay: true, Message: "History saved!" });
});

router.use("/load", (req, res) => {
    gameplayhistory = JSON.parse(fs.readFileSync("./Backend/saves/user/games_ai_history.json"));
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


export { router };
