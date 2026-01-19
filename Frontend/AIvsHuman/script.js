let xturn = true;
let round = 0;
let isAIThinking = false;

// Initialize board on page load
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById("banner").textContent = "Game Started - Your turn (X)";
    
    for (let Y = 1; Y <= 3; Y++) {
        for (let X = 1; X <= 3; X++) {
            document.getElementById("card_" + X + "-" + Y).addEventListener("click", (e) => clicked(e));
            document.getElementById("card_" + X + "-" + Y).dataset.X = X;
            document.getElementById("card_" + X + "-" + Y).dataset.Y = Y;
        }
    }
});

// Get board state as array [1 for O, -1 for X, 0 for empty]
function getBoardState() {
    const state = [];
    for (let i = 1; i <= 3; i++) {
        for (let j = 1; j <= 3; j++) {
            const cell = document.getElementById("card_" + i + "-" + j);
            if (cell.textContent === "O") {
                state.push(1);
            } else if (cell.textContent === "X") {
                state.push(-1);
            } else {
                state.push(0);
            }
        }
    }
    return state;
}

// Get available moves
function getAvailableMoves() {
    const moves = [];
    for (let i = 1; i <= 3; i++) {
        for (let j = 1; j <= 3; j++) {
            const cell = document.getElementById("card_" + i + "-" + j);
            if (cell.textContent === "") {
                const index = (j - 1) * 3 + (i - 1);
                moves.push(index);
            }
        }
    }
    return moves;
}

function clicked(e) {
    if (isAIThinking || !xturn) return;
    
    const target = e.target;
    if (target.textContent !== "") return;
    
    // Human plays X
    target.textContent = "X";
    xturn = false;
    round++;
    
    console.log(`Human played at: ${target.dataset.X}-${target.dataset.Y}`);
    
    // Check if human won
    let winner = check(target.dataset.X, target.dataset.Y);
    if (winner.wins) {
        document.getElementById("banner").textContent = "You Win!";
        banner();
        return;
    }
    
    // Check for draw
    if (round === 9) {
        document.getElementById("banner").textContent = "It's a Draw!";
        banner();
        return;
    }
    
    // AI's turn
    setTimeout(() => aiMove(), 500);
}

// Call backend API for AI move
async function aiMove() {
    if (isAIThinking) return;
    isAIThinking = true;
    
    try {
        const boardState = getBoardState();
        
        // Call backend AI API
        const response = await fetch('/api/ai/move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ board: boardState })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (!data.success) {
            console.error('AI error:', data.error);
            document.getElementById("banner").textContent = "AI Error";
            isAIThinking = false;
            return;
        }
        
        const move = data.move;
        console.log(`AI move: ${move}, confidence: ${data.confidence}`);
        
        // Convert flat index to X,Y coordinates
        const x = (move % 3) + 1;
        const y = Math.floor(move / 3) + 1;
        
        // Make the move
        const cell = document.getElementById("card_" + x + "-" + y);
        cell.textContent = "O";
        xturn = true;
        round++;
        
        // Check if AI won
        let winner = check(x, y);
        if (winner.wins) {
            document.getElementById("banner").textContent = "AI Wins!";
            banner();
            isAIThinking = false;
            return;
        }
        
        // Check for draw
        if (round === 9) {
            document.getElementById("banner").textContent = "It's a Draw!";
            banner();
            isAIThinking = false;
            return;
        }
        
        document.getElementById("banner").textContent = "Your turn (X)";
        
    } catch (error) {
        console.error('Error calling AI API:', error);
        document.getElementById("banner").textContent = "Error: " + error.message;
    }
    
    isAIThinking = false;
}

function check(X, Y) {
    const currentPlayer = xturn ? "X" : "O";
    
    const directions = [
        [1, 0],   // horizontal
        [0, 1],   // vertical
        [1, 1],   // diagonal down-right
        [1, -1]   // diagonal up-right
    ];
    
    for (let [dx, dy] of directions) {
        let count = 1;
        
        // Check in positive direction
        let x = parseInt(X) + dx;
        let y = parseInt(Y) + dy;
        while (x >= 1 && x <= 3 && y >= 1 && y <= 3) {
            let cell = document.getElementById("card_" + x + "-" + y);
            if (cell.textContent === currentPlayer) {
                count++;
            } else {
                break;
            }
            x += dx;
            y += dy;
        }
        
        // Check in negative direction
        x = parseInt(X) - dx;
        y = parseInt(Y) - dy;
        while (x >= 1 && x <= 3 && y >= 1 && y <= 3) {
            let cell = document.getElementById("card_" + x + "-" + y);
            if (cell.textContent === currentPlayer) {
                count++;
            } else {
                break;
            }
            x -= dx;
            y -= dy;
        }
        
        if (count >= 3) {
            console.log(currentPlayer + " wins!");
            return { "wins": true, "Player": currentPlayer };
        }
    }
    
    return { "wins": false, "Player": currentPlayer };
}

function banner() {
    document.getElementById("banner").classList.add("show");
}

