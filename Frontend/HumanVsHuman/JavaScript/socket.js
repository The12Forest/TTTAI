let roomId = null;
let gameDiv = document.querySelector('.site');
let gameGrid = document.getElementById('cards');
let start = null;
let opponentName = "Opponent";
let socket = io({
    path: "/socket",
    auth: {
        username: getCookie("username"),
        password: getCookie("password")
    }
});
let finished = false
setupListeners();

function updateMessage(yourTurn) {
    if (yourTurn) {
        document.querySelector(".message").textContent = "Playing against " + opponentName + " - Your turn!";
    } else {
        document.querySelector(".message").textContent = "Playing against " + opponentName + " - Wait for their move.";
    }
}

function playAgain() {
    document.getElementById("banner").classList.remove("show");
    document.getElementById("LoadingInfo").style.display = "flex";
    document.getElementById("message").classList.remove("show");
    document.getElementById('cards').classList.remove("show");

    socket.disconnect();
    socket = null;
    socket = io({
        path: "/socket",
        auth: {
            username: getCookie("username"),
            password: getCookie("password")
        }
    });

    console.log("Socket initialized");
    setBoardFromArray([0, 0, 0, 0, 0, 0, 0, 0, 0]);
    setupListeners();
    isOponentThinking = false
    finished = false
    roomId = null
    round = 0
    start = null
    xturn = true;
}

function setupListeners() {
    socket.on("startGame", (arg) => {
        console.log("Game started in room:", arg.room);
        document.getElementById("LoadingInfo").style.display = "none";
        document.querySelector(".message").classList.add("show");
    
        gameGrid.classList.add("show");
    
        console.log("New Room ID: " + arg.room)
        console.log("Opponent: " + arg.opponent)
        roomId = arg.room;
        opponentName = arg.opponent;
        start = arg.start
        if (!start) {
            isOponentThinking = true
            xturn = false;
            updateMessage(false);
        } else {
            isOponentThinking = false;
            xturn = true;
            updateMessage(true);
        }
    });
    
    socket.on("updateBoard", (arg) => {
        isOponentThinking = false;
        xturn = true; // Jetzt ist der Spieler wieder dran
        console.log("Board updated:", arg.board);
        // Update the game board with received data
        setBoardFromArray(arg.board);
        updateMessage(true);
        checkWin()
    });
    
    socket.on("playerDisconnected", (playerId) => {
        if (finished) {
            console.log("Player disconnected:", playerId);
            socket.disconnect();
            document.getElementById("outcome").textContent = "Other player disconnected!";
            banner();
        }
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