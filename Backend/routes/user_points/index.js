import express from "express";
import fs from "fs";
const router = express.Router();
const logprefix = "PlayerPoints:    ";

let users = []
let points = []

router.use("/save", (req, res) => {
    fs.writeFileSync("./Backend/saves/player-points.json", JSON.stringify(points));
    console.log(logprefix + "PlayerPoints saved:  " + JSON.stringify(points));
    res.json({ Okay: true, Message: "PlayerPoints saved!" });
});

router.use("/load", (req, res) => {
    points = JSON.parse(fs.readFileSync("./Backend/saves/player-points.json"));
    users = JSON.parse(fs.readFileSync("./Backend/saves/user/usernames.json"));
    console.log(logprefix + "Usernames loaded:     " + '["Hidden"]');
    // console.log(logprefix + "Usernames loaded:     " + JSON.stringify(users));
    console.log(logprefix + "PlayerPoints loaded:  " + JSON.stringify(points));
    res.json({ Okay: true, Message: "PlayerPoints loaded!" });
});

// Points format: [[gamesAI, winsAI], [gamesHuman, winsHuman]]
router.use("/add/account/:username", (req, res) => {
    let username = req.params.username
    if (users.indexOf(username) == -1) {
        users.push(username)
        points.push([[0, 0], [0, 0]])
        console.log(logprefix + "Account added: " + username);
        res.json({ "Okay": true });
    } else {
        console.log(logprefix + "Account already exists: " + username);
        res.status(400).json({"Okay": false, "reason": "User already exists!"})
    }
})

router.use("/remove/account/:username", (req, res) => {
    let username = req.params.username
    let userID = users.indexOf(username)
    if (userID !== -1) {
        users.splice(userID, 1)
        points.splice(userID, 1)
        console.log(logprefix + "Account removed: " + username);
        res.json({ "Okay": true });
    } else {
        console.log(logprefix + "Account not found: " + username);
        res.status(400).json({ "Okay": false, "reason": "User does not exist!" })
    }
})

router.use("/countAI/:user/:wone", async (req, res) => {
    let userID = users.indexOf(req.params.user)
    console.log(logprefix + JSON.stringify(users))
    if (userID !== -1) {
        points[userID][0][0]++;  // gamesAI
        if(req.params.wone == 1){
            points[userID][0][1]++;  // winsAI
        }
        console.log(logprefix + "AI game recorded for " + req.params.user + " (won: " + req.params.wone + ")");
        res.json({ "Okay": true });
    } else {
        console.log(logprefix + "AI game failed - user not found: " + req.params.user);
        res.status(400).json({ "Okay": false, "Reason": "User does not exist!" });
    }
})

router.use("/countHuman/:user/:wone", async (req, res) => {
    let userID = users.indexOf(req.params.user)
    if (userID !== -1) {
        points[userID][1][0]++;  // gamesHuman
        if (req.params.wone == 1) {
            points[userID][1][1]++;  // winsHuman
        }
        console.log(logprefix + "Human game recorded for " + req.params.user + " (won: " + req.params.wone + ")");
        res.json({ "Okay": true });
    } else {
        console.log(logprefix + "Human game failed - user not found: " + req.params.user);
        res.status(400).json({ "Okay": false, "Reason": "User does not exist!" });
    }
})

router.use("/getStats/:user", async (req, res) => {
    let userID = users.indexOf(req.params.user)
    if (userID !== -1) {
        const userPoints = points[userID];
        // Format: [[gamesAI, winsAI], [gamesHuman, winsHuman]]
        const aiGames = userPoints[0][0];
        const aiWins = userPoints[0][1];
        const humanGames = userPoints[1][0];
        const humanWins = userPoints[1][1];
        
        console.log(logprefix + "Stats requested for " + req.params.user);
        res.json({
            "Okay": true,
            "ai": {
                "games": aiGames,
                "wins": aiWins,
                "losses": aiGames - aiWins,
                "ratio": aiGames > 0 ? (aiWins / aiGames).toFixed(2) : 0
            },
            "human": {
                "games": humanGames,
                "wins": humanWins,
                "losses": humanGames - humanWins,
                "ratio": humanGames > 0 ? (humanWins / humanGames).toFixed(2) : 0
            }
        });
    } else {
        console.log(logprefix + "Stats failed - user not found: " + req.params.user);
        res.status(400).json({ "Okay": false, "Reason": "User does not exist!" });
    }
})

export { router };