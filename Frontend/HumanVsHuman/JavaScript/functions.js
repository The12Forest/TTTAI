// Check if a player has won
function checkWin() {
    if (start) {
        let winner = checkPlayer("X");
        if (winner) {
            document.getElementById("outcome").textContent = "You Win!";
            updateMiniBoard();
            document.getElementById("banner").classList.add("show");;
            finished = true
            send_history(true)
            return;
        }
        winner = checkPlayer("O");
        if (winner) {
            document.getElementById("outcome").textContent = "You have lost!";
            updateMiniBoard();
            document.getElementById("banner").classList.add("show");
            finished = true
            send_history(false)
            return;
        }
    } else {
        let winner = checkPlayer("O");
        if (winner) {
            document.getElementById("outcome").textContent = "You Win!";
            updateMiniBoard();
            document.getElementById("banner").classList.add("show");
            finished = true
            send_history(true)
            return;
        }
        winner = checkPlayer("X");
        if (winner) {
            document.getElementById("outcome").textContent = "You have lost!";
            updateMiniBoard();
            document.getElementById("banner").classList.add("show");
            finished = true
            send_history(false)
            return;
        }
    }
    if (round === 9) {
        document.getElementById("outcome").textContent = "It's a Draw!";
        updateMiniBoard();
        document.getElementById("banner").classList.add("show");
        finished = true
        return;
    }
}

function updateMiniBoard() {
    const miniCells = document.querySelectorAll('.mini-cell');
    for (let i = 0; i < 9; i++) {
        const cell = getCell(i);
        miniCells[i].textContent = cell.textContent;
    }
}

function send_history(hasWon) {
    let hasWoneString
    if (hasWon) {hasWoneString = 1} else {hasWoneString = 0}
    fetch("/api/points/countHuman/" + encodeURIComponent(username) + "/" + hasWoneString)
}

window.addEventListener("unload", () => {
    if (finished) return;

    const url = `/api/points/countHuman/${encodeURIComponent(username)}/0`;
    navigator.sendBeacon(url);
});
  