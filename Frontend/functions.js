// Check if a player has won
function checkPlayer(player) {
    for (const line of WIN_LINES) {
        if (line.every(i => getCell(i).textContent === player)) {
            console.log(player + " wins!");
            return true;
        }
    }
    return false;
}

function getCookie(name) {
    return document.cookie
        .split("; ")
        .find(row => row.startsWith(name + "="))
        ?.split("=")[1];
}

function capitalizeFirstLetter(str) {
    if (!str) return str;
    return str.charAt(0).toUpperCase() + str.slice(1);
}


async function sha256(message) {
    const msgBuffer = new TextEncoder().encode(message);
    const hashBuffer = await crypto.subtle.digest("SHA-256", msgBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, "0")).join("");
    return hashHex;
}

async function login() {
    try {
        const res = await fetch(
            "/api/user/login/" +
            encodeURIComponent(getCookie("username")) +
            "/" +
            encodeURIComponent(getCookie("password"))
        );

        const data = await res.json();

        if (data.Okay) {
            return true;
        } else {
            window.location.href = "/login";
            return false;
        }
    } catch (err) {
        console.error(err);
        return false;
    }
}