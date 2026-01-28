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
    if (users.indexOf(username) === -1) {
        users.push(username)
        points.push([[0, 0], [0, 0]])
        res.json({ "Okay": true });
    } else {
        res.status(400).json({"Okay": false, "reason": "User already exists!"})
    }
})

router.use("/remove/account/:username", (req, res) => {
    let username = req.params.username
    let userID = users.indexOf(username)
    if (userID !== -1) {
        users.splice(userID, 1)
        points.splice(userID, 1)
        res.json({ "Okay": true });
    } else {
        res.status(400).json({ "Okay": false, "reason": "User does not exist!" })
    }
})

router.use("/countAI/:user/:wone", async (req, res) => {
    let userID = users.indexOf(req.params.user)
    if (userID !== -1) {
        points[userID][0][0]++;  // gamesAI
        if(req.params.wone == 1){
            points[userID][0][1]++;  // winsAI
        }
        res.json({ "Okay": true });
    } else {
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
        res.json({ "Okay": true });
    } else {
        res.status(400).json({ "Okay": false, "Reason": "User does not exist!" });
    }
})

export { router };