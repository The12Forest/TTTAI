let xturn = true;
let round = 0;
let isAIThinking = false;

// Win lines as index arrays
const WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8], // rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8], // cols
    [0, 4, 8], [2, 4, 6]             // diagonals
];

// Get cell by index 0-8
function getCell(index) {
    return document.getElementById("card_" + index);
}

// Initialize board on page load
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById("banner").textContent = "Game Started - Your turn (X)";
    
    for (let i = 0; i < 9; i++) {
        const cell = getCell(i);
        cell.dataset.index = i;
        cell.addEventListener("click", (e) => clicked(e));
    }
});

// Get board state as array [-1 for O, 1 for X, 0 for empty]
function getBoardState() {
    const state = [];
    for (let i = 0; i < 9; i++) {
        const cell = getCell(i);
        if (cell.textContent === "O") {
            state.push(-1);
        } else if (cell.textContent === "X") {
            state.push(1);
        } else {
            state.push(0);
        }
    }
    return state;
}

// Set the entire board from array [-1, 0, 1]
function setBoardFromArray(arr) {
    round = 0;
    for (let i = 0; i < 9; i++) {
        const cell = getCell(i);
        if (arr[i] === -1) {
            cell.textContent = "O";
            round++;
        } else if (arr[i] === 1) {
            cell.textContent = "X";
            round++;
        } else {
            cell.textContent = "";
        }
    }
}

function clicked(e) {
    if (isAIThinking || !xturn) return;
    
    const target = e.target;
    if (target.textContent !== "") return;
    
    const index = parseInt(target.dataset.index);
    
    // Human plays X
    target.textContent = "X";
    xturn = false;
    round++;
    
    console.log(`Human played at index: ${index}`);
    
    // Check if human won
    checkWin()
    
    // Check for draw
    if (round === 9) {
        document.getElementById("banner").textContent = "It's a Draw!";
        banner();
        return;
    }
    
    // AI's turn
    setTimeout(() => aiMove(), 10);
}

// Call backend API for AI move
async function aiMove() {
    if (isAIThinking) return;
    isAIThinking = true;
    
    try {
        const boardState = getBoardState();
        
        // Call backend AI API
        const response = await fetch('/api/ai/getAIMove?Board=' + JSON.stringify(boardState));
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (!data.Okay) {
            console.error('AI error:', data.error);
            document.getElementById("banner").textContent = "AI Error";
            isAIThinking = false;
            return;
        }

        const newBoard = data.Borad || data.Board;
        
        // Find AI move by comparing boards
        let aiMoveIndex = 1;
        for (let i = 0; i < 9; i++) {
            if (boardState[i] !== newBoard[i] && newBoard[i] === 1) {
                aiMoveIndex = i;
                break;
            }
        }
        
        // Set entire board from response
        setBoardFromArray(newBoard);
        xturn = true;
        
        console.log(`AI played at index: ${aiMoveIndex}`);
        
        // Check if AI won
        checkWin()
        
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

// Check if a player has won
function checkWin() {
    if (isAIThinking) {
        let winner = checkPlayer("X");
        if (winner) {
            document.getElementById("banner").textContent = "You Win!";
            banner();
            finish = true
            return;
        }
        winner = checkPlayer("O");
        if (winner) {
            document.getElementById("banner").textContent = "You have lost!";
            banner();
            finish = true
            return;
        }
    } else {
        let winner = checkPlayer("O");
        if (winner) {
            document.getElementById("banner").textContent = "You Win!";
            banner();
            finish = true
            return;
        }
        winner = checkPlayer("X");
        if (winner) {
            document.getElementById("banner").textContent = "You have lost!";
            banner();
            finish = true
            return;
        }
    }
}

function checkPlayer(player) {
    for (const line of WIN_LINES) {
        if (line.every(i => getCell(i).textContent === player)) {

            console.log(player + " wins!");
            return true;
        }
    }
    return false;
}

function banner() {
    document.getElementById("banner").classList.add("show");
}

