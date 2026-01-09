import express from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
const router = express.Router();
const logprefix = "LoginRouter:     ";

let passwords = [];
let usernames = [];
let sessions = [];
// let passwordreset = [];

const adminPanelDir = "../../../Frontend/admin";
const adminPanelPath = path.join(adminPanelDir, "main.html");

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// function makeid(length) {
//   var result = "";
//   var characters =
//     "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
//   var charactersLength = characters.length;
//   for (var i = 0; i < length; i++) {
//     result += characters.charAt(Math.floor(Math.random() * charactersLength));
//   }
//   return result;
// }

router.use("/save", (req, res) => {
    fs.writeFileSync("./Backend/saves/user/usernames.json", JSON.stringify(usernames));
    fs.writeFileSync("./Backend/saves/user/passwords.json", JSON.stringify(passwords));
    console.log(logprefix + "Username saved:      " + '["Hidden"]');
    console.log(logprefix + "Passwords saved:     " + '["Hidden"]');
    // console.log(logprefix + "Usernames saved:     " + JSON.stringify(usernames));
    // console.log(logprefix + "Passwords saved:     " + JSON.stringify(passwords));
    res.json({ Okay: true, Message: "Users saved!" });
});

router.use("/load", (req, res) => {
    usernames = JSON.parse(fs.readFileSync("./Backend/saves/user/usernames.json"));
    passwords = JSON.parse(fs.readFileSync("./Backend/saves/user/passwords.json"));
    console.log(logprefix + "Usernames loaded:     " + '["Hidden"]');
    console.log(logprefix + "Passwords loaded:     " + '["Hidden"]');
    // console.log(logprefix + "Usernames loaded:  " + JSON.stringify(usernames))
    // console.log(logprefix + "Passwords loaded:  " + JSON.stringify(passwords))
    res.json({ Okay: true, Message: "Users loaded!" });
});

router.get("/login/:username/:passwd", (req, res) => {
    const username = req.params.username;
    const passwd = req.params.passwd;

    let userIndex = usernames.indexOf(username);

    if (userIndex !== -1 && passwords[userIndex] === passwd) {
        console.log(logprefix + "User '" + username + "' logged in successfully.");
        res.json({ Okay: true, Message: "Login successful!" });
    } else {
        console.log(logprefix + "User '" + username + "' login failed - wrong credentials.");
        res.status(401).json({ Okay: false, Reason: "Invalid username or password" });
    }
});

router.get("/register/:username/:passwd", (req, res) => {
    const username = req.params.username;
    const passwd = req.params.passwd;

    let userIndex = usernames.indexOf(username);
    let passIndex = passwords.indexOf(passwd);

    if (userIndex === -1) {
        if (passIndex === -1) {
            usernames.push(username);
            passwords.push(passwd);

            console.log(logprefix + "User '" + username + "' registered successfully.");
            res.json({ Okay: true, Reason: "User created successfully!" });
        } else {
            console.log(logprefix + "Registration failed for '" + username + "' - password already in use.");
            res.json({ Okay: false, Reason: "Password already in use!" });
        }
    } else {
        console.log(logprefix + "Registration failed - user '" + username + "' already exists.");
        res.status(100).json({ Okay: false, Reason: "User already exists!" });
    }
});


router.get("/delete/:username/:passwd", (req, res) => {
    let UuserID = passwords.indexOf(req.params.username)
    let PuserID = passwords.indexOf(req.params.passwd)
    if (UuserID === PuserID && PuserID !== -1) {
        usernames.splice(req.params.username, 1)
        passwords.splice(req.params.passwd, 1)
        res.json({"Okay": true, "Reason": "Deleted user successfully!"})
    } else {
        res.json({ "Okay": false, "Reason": "User dose not exist!" })
    }
});

router.get("/changepasswd/:username/:passwdold/:passwdnew", (req, res) => {
    let UuserID = passwords.indexOf(req.params.username)
    let PuserID = passwords.indexOf(req.params.passwdold)
    if (UuserID === PuserID && PuserID !== -1) {
        passwords[UuserID] = req.params.passwdnew
        res.json({ "Okay": true, "Reason": "Changed password successfully!" })
    } else {
        res.json({ "Okay": false, "Reason": "User dose not exist or the Login was wrong!" })
    }
});

router.get("/check/:username/:passwd", (req, res) => {
    let UuserID = passwords.indexOf(req.params.username)
    let PuserID = passwords.indexOf(req.params.passwd)
    if (UuserID === PuserID && PuserID !== -1) {
        res.json({ Okay: true });
    } else {
        res.send({ Okay: false });
    }
});


router.use("", (req, res) => res.status(404).json({ error: "not found" }));
export { router };
