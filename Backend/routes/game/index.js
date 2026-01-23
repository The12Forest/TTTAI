import express from "express";
import { Server } from "socket.io";

const router = express.Router();
const logprefix = "GameRouter:      ";

// let sessions = [];
let io = null;

// Spiel-Logik
let waitingPlayer = null;
const sessions = {}; // roomId → gameState

// Function to initialize socket.io with the HTTP server
export function initializeSocket(httpServer) {
    console.log(logprefix)
    io = new Server(httpServer, {
        cors: { origin: "*" }, // nur für Test, in Produktion anpassen
        path: "/socket" // Custom path for socket.io
    });

    io.on("connection", (socket) => {
        console.log(logprefix + "Spieler verbunden:", socket.id);

        if (waitingPlayer) {
            // Raum erstellen
            const room = `game-${socket.id}-${waitingPlayer.id}`;
            socket.join(room);
            waitingPlayer.join(room);

            // Game State initialisieren
            sessions[room] = {
                players: [waitingPlayer.id, socket.id],
                board: [0,0,0, 0,0,0, 0,0,0]
            };

            // Start-Event senden
            io.to(socket.id).emit("startGame", { "room": room, "start": false });
            io.to(waitingPlayer.id).emit("startGame", { "room": room, "start": true });


            waitingPlayer = null;
        } else {
            waitingPlayer = socket;
        }

        socket.on("playerMove", (arg) => {
            if (!sessions[arg.room]) return;
            let players = sessions[arg.room].players;

            console.log(logprefix + "playerMove received from: " + socket.id);

            if (players[0] === socket.id) {
                io.to(players[1]).emit("updateBoard", { room: arg.room, board: arg.board });
                console.log(logprefix + "Sent to player 1: " + players[1]);
            } else if (players[1] === socket.id) {
                io.to(players[0]).emit("updateBoard", { room: arg.room, board: arg.board });
                console.log(logprefix + "Sent to player 0: " + players[0]);
            }
        });

        socket.on("disconnect", () => {
            if (waitingPlayer === socket) waitingPlayer = null;

            // Räume aufräumen
            for (const room of Object.keys(sessions)) {
                if (sessions[room].players.includes(socket.id)) {
                    io.to(room).emit("playerDisconnected", socket.id);
                    delete sessions[room];
                }
            }

            console.log(logprefix + "Spieler getrennt:", socket.id);
        });
    });

    console.log(logprefix + "Socket.IO initialized at /api/game/socket");
    return io;
}

// Route to check socket status
router.get('/socket/status', (req, res) => {
    res.json({ 
        status: io ? 'active' : 'inactive',
        path: '/api/game/socket',
        message: 'Socket.IO für TTTAI'
    });
});

export { router };
