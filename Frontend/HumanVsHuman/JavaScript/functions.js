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
  