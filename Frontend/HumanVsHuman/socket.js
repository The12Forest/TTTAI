const socket = io({
    path: "/socket"
});
let roomId = null;
let gameDiv = document.querySelector('.site');
let gameGrid = document.getElementById('cards');

// window.addEventListener("DOMContentLoaded", () => {

// });
let start = null
socket.on("startGame", (arg) => {
    console.log("Game started in room:", arg.room);
    document.getElementById("LoadingInfo").style.display = "none";
    gameGrid.classList.toggle("show")

    roomId = arg.room;
    start = arg.start
    if (!start) {
        isOponentThinking = true
    }
});

socket.on("updateBoard", (arg) => {
    isOponentThinking = false;
    xturn = true; // Jetzt ist der Spieler wieder dran
    console.log("Board updated:", arg.board);
    // Update the game board with received data
    setBoardFromArray(arg.board);
    checkWin()
});

socket.on("playerDisconnected", (playerId) => {
    console.log("Player disconnected:", playerId);
    if (!finish) {
        alert("Other player disconnected!");
    }
});

// Function to send move to other player
function sendMove(board) {
    isOponentThinking = true
    if (roomId) {
        socket.emit("playerMove", { "room": roomId, "board": board });
    }     
}


console.log("Socket initialized");



// Ich hasse Git