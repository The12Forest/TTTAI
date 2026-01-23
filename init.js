import express from "express";
import path from "path";
import fs from "fs";
import https from "https";
import http from "http";
import { fileURLToPath } from "url";
const baseurl = "http://127.0.0.1";

//Router Laden
import { router as userRouter } from "./Backend/routes/user/index.js";
import { router as mainRouter } from "./Backend/routes/main/index.js";
// import { router as aiRouter } from "./Backend/routes/ai/index.js";
// import { router as adminRouter } from "./Backend/routes/storage/index.js";
// import { router as loginRouter } from "./Backend/routes/login/index.js";
// import { router as shutdownRouter } from "./Backend/routes/shutdown/index.js";

//Setting Variables
const app = express();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

//Ports Definieren
const httpPort = 80;
const httpsPort = 443;

// SSL-Zertifikate laden
const privateKey = fs.readFileSync("./cert/key.pem", "utf8");
const certificate = fs.readFileSync("./cert/cert.pem", "utf8");
const credentials = { key: privateKey, cert: certificate };

// Custom File Mappings
// app.get("/", (req, res) => {
//   res.sendFile(__dirname + "/Frontend/home/index.html");
// });

// Foldermapings
app.use("/", express.static("./Frontend/AIvsHuman"));
app.use("/login", express.static("./Frontend/login"));
app.use("/register", express.static("./Frontend/register"));
app.use("/panel", express.static("./Frontend/panel"));
app.use("/play/ai", express.static("./Frontend/AIvsHuman"));
app.use("/play/vs", express.static("./Frontend/HumanVsHuman"));
app.use("/log", express.static("./LOG"));

app.use('/api/ai/models', express.static("./Backend/routes/ai/models"));

// Routerroutes
app.use("/api/user", userRouter);
app.use("/api/main", mainRouter);
// app.use("/api/ai", aiRouter);
// app.use("/api/user", userRouter);
// app.use("/api/storage", adminRouter);
// app.use("/api/login", loginRouter);
// app.use("/api/shutdown", shutdownRouter);



// app.get("/Main", (req, res) => {
//   res.redirect("/");
// });
// app.use("", (req, res) => {
//   res.redirect("/");
// });

//HTTP-Server
http.createServer(app).listen(httpPort, '0.0.0.0', () => {
  console.log(`HTTP server running on port ${httpPort}`);
});

// HTTPS-Server
https.createServer(credentials, app).listen(httpsPort, '0.0.0.0', () => {
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

// console.clear();
console.log("Server Startup!");
// fetch(baseurl + "/api/main/load");
// fetch(baseurl + "/api/ai/load");


export default app;


