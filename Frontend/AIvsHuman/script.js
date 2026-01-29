let xturn = true;
let round = 0;
let isAIThinking = false;
let finished = false
let game_play_history = []
const username = getCookie("username");
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
    for (let i = 0; i < 9; i++) {
        const cell = getCell(i);
        cell.dataset.index = i;
        cell.addEventListener("click", (e) => clicked(e));
    }
    document.getElementById("playagain").addEventListener("click", () => {
        playAgain();
    });
    document.getElementById("GoToPanel").addEventListener("click", () => {
        window.location = "/panel";
    });

    
    if (!username) {
        window.location.href = "/login";
    }

});

function playAgain() {
    document.getElementById("banner").classList.remove("show");
    // document.getElementById("message").classList.remove("show");
    document.getElementById('cards').classList.remove("show");

    setBoardFromArray([0, 0, 0, 0, 0, 0, 0, 0, 0]);
    isOponentThinking = false
    round = 0
    xturn = true;
    game_play_history = []
    finished = false
}

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

    for (let i = 0; i < 9; i++) {
        const cell = getCell(i);
        if (arr[i] === -1) {
            cell.textContent = "O";
        } else if (arr[i] === 1) {
            cell.textContent = "X";
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
    
    game_play_history.push(getBoardState())
    checkWin();
    
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
            banner();
            document.getElementById("outcome").textContent = "AI Error";
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
        round++;
        xturn = true;
        
        console.log(`AI played at index: ${aiMoveIndex}`);
        
        // Check if AI won
        game_play_history.push(getBoardState())
        checkWin();
                
    } catch (error) {
        console.error('Error calling AI API:', error);
        document.getElementById("outcome").textContent = "Error: " + error.message;
        banner();
    }
    
    isAIThinking = false;
}

function checkWin() {
    let winner = checkPlayer("X");
    if (winner) {
        document.getElementById("outcome").textContent = "You Win!";
        banner();
        send_play_history()
        send_history(true)
        finished = true
        return;
    }

    winner = checkPlayer("O");
    if (winner) {
        document.getElementById("outcome").textContent = "AI has won";
        banner();
        send_history(false)
        finished = true
        return;
    }

    if (round === 9) {
        document.getElementById("outcome").textContent = "It's a Draw!";
        banner();
        finished = true
        return;
    }
    isAIThinking = false;
}

// Check if a player has won
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

function send_play_history() {
    fetch('/api/ai/gameplayhistory', {
        method: 'POST', // or PUT, PATCH
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            gameplay: game_play_history
        }),
    })
        .then(res => {
            if (!res.ok) throw new Error('Request failed');
            return res.json();
        })
        .catch(err => console.error(err));
}

function send_history(hasWone) {
    let hasWoneString
    if (hasWone) { hasWoneString = 1 } else { hasWoneString = 0 }
    fetch("/api/points/countAI/" + username + "/" + hasWoneString)
}

window.addEventListener("unload", () => {
    if (finished) return;

    const url = `/api/points/countAI/${username}/0`;
    navigator.sendBeacon(url);
});
