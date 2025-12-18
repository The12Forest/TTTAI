import express from "express";
import fs from "fs";
import path from "path";
import { buffer } from "stream/consumers";
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

function allowLocalhostOnly(req, res, next) {
    const ip = req.ip || req.connection.remoteAddress;

    if (ip === '127.0.0.1' || ip === '::1') {
        return next();
    } else {
        res.status(403).send('Forbidden: Only localhost allowed');
    }
}

router.use("/save", (req, res) => {
    fs.writeFileSync("./Backend/saves/user/passwords.json", JSON.stringify(passwords));
    fs.writeFileSync("./Backend/saves/admin_usernames.json", JSON.stringify(usernames));
    console.log(logprefix + "Usernames saved:     " + JSON.stringify(usernames));
    console.log(logprefix + "Passwords saved:     " + JSON.stringify(passwords));
    //   console.log(logprefix + "Username saved:      " + '["Hidden"]');
    //   console.log(logprefix + "Passwords saved:     " + '["Hidden"]');
    res.json({ Okay: true, Message: "Users saved!" });
});

router.use("/load", (req, res) => {
    usernames = JSON.parse(fs.readFileSync("./Backend/saves/admin_usernames.json"));
    passwords = JSON.parse(fs.readFileSync("./Backend/saves/passwords.json"));
    console.log(logprefix + "Usernames loaded:     " + '["Hidden"]');
    console.log(logprefix + "Passwords loaded:     " + '["Hidden"]');
    // console.log(logprefix + "Usernames loaded:  " + JSON.stringify(usernames))
    // console.log(logprefix + "Passwords loaded:  " + JSON.stringify(passwords))
    res.json({ Okay: true, Message: "Users loaded!" });
});

router.get("/create/:username/:passwd", (req, res) => {
    let UuserID = passwords.indexOf(req.params.username)
    let PuserID = passwords.indexOf(req.params.passwd)
    if (UuserID === -1) {
        if (PuserID === -1) {
            usernames.push(req.params.username)
            passwords.push(req.params.passwd)
            res.json({"Okay": true, "Reason": "User created successfully!"})
        } else {
            res.json({"Okay": false, "Reason": "Useralready exists!"})
        }
    } else {
        res.json({"Okay": false, "Reason": "User already exists!"})
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

router.get("/:passwd", (req, res) => {
    let passwdIndex = passwords.indexOf(req.params.passwd);

    if (passwdIndex !== -1) {
        console.log(logprefix + "Admin user: " + passwdIndex + " logged in.");
        res.sendFile(path.resolve(__dirname, adminPanelPath));
    } else {
        console.log(
            logprefix +
            "Admin user tried to login but the username/password was wrong."
        );
        res.status(401).send({ Okay: false, Reason: "Unauthorized" });
    }
});

router.use("", (req, res) => res.status(404).json({ error: "not found" }));
export { router };
