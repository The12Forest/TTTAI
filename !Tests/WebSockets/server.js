import { createServer } from "http";
import { Server } from "socket.io";

const httpServer = createServer();
const io = new Server(httpServer, {
    cors: { origin: "*" } // nur für Test, in Produktion anpassen
});

// Spiel-Logik
let waitingPlayer = null;
const games = {}; // roomId → gameState

io.on("connection", (socket) => {
    console.log("Spieler verbunden:", socket.id);

    if (waitingPlayer) {
        // Raum erstellen
        const room = `game-${socket.id}-${waitingPlayer.id}`;
        socket.join(room);
        waitingPlayer.join(room);

        // Game State initialisieren
        games[room] = {
            players: [waitingPlayer.id, socket.id],
            scores: { [waitingPlayer.id]: 0, [socket.id]: 0 }
        };

        // Start-Event senden
        io.to(room).emit("startGame", { room, players: games[room].players, scores: games[room].scores });

        waitingPlayer = null;
    } else {
        waitingPlayer = socket;
        socket.emit("waitingForOpponent");
    }

    socket.on("playerMove", ({ room }) => {
        if (!games[room]) return;

        games[room].scores[socket.id] += 1;

        // Scores an beide Spieler senden
        io.to(room).emit("updateScores", games[room].scores);
    });

    socket.on("disconnect", () => {
        if (waitingPlayer === socket) waitingPlayer = null;

        // Räume aufräumen
        for (const room of Object.keys(games)) {
            if (games[room].players.includes(socket.id)) {
                io.to(room).emit("playerDisconnected", socket.id);
                delete games[room];
            }
        }

        console.log("Spieler getrennt:", socket.id);
    });
});

httpServer.listen(3000, () => {
    console.log("Server läuft auf Port 3000");
});
