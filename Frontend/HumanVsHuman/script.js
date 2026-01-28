let xturn = true;
let round = 0;
let isOponentThinking = false;

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
    if (isOponentThinking || !xturn) return;

    const target = e.target;
    if (target.textContent !== "") return;

    const index = parseInt(target.dataset.index);

    // Human plays X
    if (start) {
        target.textContent = "X";
    } else {
        target.textContent = "O";
    }

    xturn = false;
    round++;

    console.log(`Human played at index: ${index}`);

    sendMove(getBoardState());

    checkWin()
}

// Check if a player has won
function checkWin() {
    if (start) {
        let winner = checkPlayer("X");
        if (winner) {
            document.getElementById("outcome").textContent = "You Win!";
            banner();
            return;
        }
        winner = checkPlayer("O");
        if (winner) {
            document.getElementById("outcome").textContent = "You have lost!";
            banner();
            return;
        }
    } else {
        let winner = checkPlayer("O");
        if (winner) {
            document.getElementById("outcome").textContent = "You Win!";
            banner();
            return;
        }
        winner = checkPlayer("X");
        if (winner) {
            document.getElementById("outcome").textContent = "You have lost!";
            banner();
            return;
        }
    }
    if (round === 9) {
        document.getElementById("banner").textContent = "It's a Draw!";
        banner();
        return;
    }
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

