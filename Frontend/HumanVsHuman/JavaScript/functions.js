function getCookie(name) {
    return document.cookie
        .split("; ")
        .find(row => row.startsWith(name + "="))
        ?.split("=")[1];
}

// Check if a player has won
function checkWin() {
    if (start) {
        let winner = checkPlayer("X");
        if (winner) {
            document.getElementById("outcome").textContent = "You Win!";
            document.getElementById("banner").classList.add("show");;
            finished = true
            send_history(true)
            return;
        }
        winner = checkPlayer("O");
        if (winner) {
            document.getElementById("outcome").textContent = "You have lost!";
            document.getElementById("banner").classList.add("show");
            finished = true
            send_history(false)
            return;
        }
    } else {
        let winner = checkPlayer("O");
        if (winner) {
            document.getElementById("outcome").textContent = "You Win!";
            document.getElementById("banner").classList.add("show");
            finished = true
            send_history(true)
            return;
        }
        winner = checkPlayer("X");
        if (winner) {
            document.getElementById("outcome").textContent = "You have lost!";
            document.getElementById("banner").classList.add("show");
            finished = true
            send_history(false)
            return;
        }
    }
    if (round === 9) {
        document.getElementById("outcome").textContent = "It's a Draw!";
        document.getElementById("banner").classList.add("show");
        finished = true
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

function send_history(hasWon) {
    let hasWoneString
    if (hasWon) {hasWoneString = 1} else {hasWoneString = 0}
    fetch("/api/points/countHuman/" + username + "/" + hasWoneString)
}

window.addEventListener("unload", () => {
    if (finished) return;

    const url = `/api/points/countHuman/${username}/0`;
    navigator.sendBeacon(url);
});
  