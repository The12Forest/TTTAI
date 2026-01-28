let roomId = null;
let gameDiv = document.querySelector('.site');
let gameGrid = document.getElementById('cards');
let start = null
let socket = io({
    path: "/socket"
});
setupListeners();

function playAgain() {
    document.getElementById("banner").classList.remove("show");
    document.getElementById("LoadingInfo").style.display = "flex";
    document.querySelector(".message").classList.remove("show");
    gameGrid.classList.remove("show");

    socket.disconnect();
    socket = null;
    socket = io({
        path: "/socket"
    });

    console.log("Socket initialized");
    setBoardFromArray([0, 0, 0, 0, 0, 0, 0, 0, 0]);
    setupListeners();
    isOponentThinking = false
    roomId = null
    round = 0
    start = null
    xturn = false;
}

function setupListeners() {
    socket.on("startGame", (arg) => {
        console.log("Game started in room:", arg.room);
        document.getElementById("LoadingInfo").style.display = "none";
        document.querySelector(".message").classList.add("show");
    
        gameGrid.classList.add("show");
    
        console.log("New Room ID: " + arg.room)
        roomId = arg.room;
        start = arg.start
        if (!start) {
            document.querySelector(".message").textContent = "Wait for the oponenet to make there first move."
            isOponentThinking = true
        } else {
            document.querySelector(".message").textContent = "You can make the first Move!"
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
        socket.disconnect();
        document.getElementById("outcome").textContent = "Other player disconnected!";
        banner();
    });
}

// Function to send move to other player
function sendMove(board) {
    isOponentThinking = true
    if (roomId) {
        socket.emit("playerMove", { "room": roomId, "board": board });
    }     
}





// Ich hasse Git