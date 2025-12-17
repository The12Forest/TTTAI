import express from "express";
import path from "path";
import fs from "fs";
import https from "https";
import http from "http";
import { fileURLToPath } from "url";
const baseurl = "http://127.0.0.1";

//Router Laden
import { router as gamingtimeRouter } from "./Backend/routes/time/index.js";
import { router as tasksRouter } from "./Backend/routes/tasks/index.js";
import { router as userRouter } from "./Backend/routes/user/index.js";
import { router as adminRouter } from "./Backend/routes/storage/index.js";
import { router as loginRouter } from "./Backend/routes/login/index.js";
import { router as shutdownRouter } from "./Backend/routes/shutdown/index.js";

//Setting Variables
const app = express();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

//Ports Definieren
const httpPort = 80;
const httpsPort = 10108;

// SSL-Zertifikate laden
const privateKey = fs.readFileSync("./Cert/key.pem", "utf8");
const certificate = fs.readFileSync("./Cert/cert.pem", "utf8");
const credentials = { key: privateKey, cert: certificate };

// Custom File Mappings
app.get("/", (req, res) => {
  res.sendFile(__dirname + "/Frontend/main/main.html");
});
app.get("/user", (req, res) => {
  res.sendFile(__dirname + "/Frontend/user/main.html");
});
app.get("/admin", (req, res) => {
  res.sendFile(__dirname + "/Frontend/admin_login/main.html");
});
app.get("/admin-control/style.css", (req, res) => {
  res.sendFile(__dirname + "/Frontend/admin/style.css");
});
app.get("/admin-control/script.js", (req, res) => {
  res.sendFile(__dirname + "/Frontend/admin/script.js");
});

// Foldermapings
app.use("/", express.static("./Frontend/main"));
app.use("/user", express.static("./Frontend/user"));
app.use("/admin", express.static("./Frontend/admin_login"));

// Routerroutes
app.use("/api/time", gamingtimeRouter);
app.use("/api/task", tasksRouter);
app.use("/api/user", userRouter);
app.use("/api/storage", adminRouter);
app.use("/api/login", loginRouter);
app.use("/api/shutdown", shutdownRouter);

app.get("/Main", (req, res) => {
  res.redirect("/");
});
app.use("", (req, res) => {
  res.redirect("/");
});

//HTTP-Server
http.createServer(app).listen(httpPort, () => {
  console.log(`HTTP server running on port ${httpPort}`);
});

// HTTPS-Server
https.createServer(credentials, app).listen(httpsPort, () => {
  console.log(`HTTPS server running on port ${httpsPort}`);
});

// Log
function Time() {
  const now = new Date();

  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  const hours = String(now.getHours()).padStart(2, "0");
  const minutes = String(now.getMinutes()).padStart(2, "0");
  const seconds = String(now.getSeconds()).padStart(2, "0");
  const milliseconds = String(now.getMilliseconds()).padStart(3, "0");

  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}.${milliseconds}      `;
}

const originalLog = console.log;

if (!fs.existsSync("./LOG")) {
  fs.mkdirSync("./LOG", { recursive: true });
}

console.log = function (message, ...optionalParams) {
  fs.createWriteStream(
    "./LOG/LOG_" +
      new Date().getFullYear() +
      "-" +
      String(new Date().getMonth() + 1).padStart(2, "0") +
      "-" +
      String(new Date().getDate()).padStart(2, "0") +
      ".log",
    { flags: "a" }
  ).write(Time() + message + " " + optionalParams.join(" ") + "\n");
  originalLog(Time() + message, ...optionalParams);
};

console.log("Server Startup!");
fetch(baseurl + "/api/storage/load");

export default app;
