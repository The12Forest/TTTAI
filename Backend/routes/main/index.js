import express from "express";
const router = express.Router();
const logprefix = "MainRouter:      ";
const baseurl = "http://localhost:80/api";

router.use("/save", (req, res) => {
    fetch(baseurl + "/user/save")
    fetch(baseurl + "/ai/save")
    fetch(baseurl + "/points/save")
    // fetch(baseurl + "/router/save")
    res.json({ Okay: true, Message: "Everything saved succesfully!" });
});

router.use("/load", (req, res) => {
    fetch(baseurl + "/user/load")
    fetch(baseurl + "/ai/load")
    fetch(baseurl + "/points/load")
    // fetch(baseurl + "/router/load")
    res.json({ Okay: true, Message: "Everything loaded succesfully!" });
});


router.use("", (req, res) => res.status(404).json({ error: "not found" }));
export { router };