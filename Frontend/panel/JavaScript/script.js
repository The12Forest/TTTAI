const username = getCookie("username");
document.getElementById("username").innerText = capitalizeFirstLetter(username);
if (!username) {
    window.location.href = "/login";
}

async function loadStats() {
    try {
        const res = await fetch("/api/points/getStats/" + encodeURIComponent(username));
        const data = await res.json();
        if (data.Okay) {
            document.getElementById("ai-ratio").textContent = data.ai.wins + "/" + data.ai.losses;
            document.getElementById("human-ratio").textContent = data.human.wins + "/" + data.human.losses;
        }
    } catch (err) {
        console.error("Failed to load stats:", err);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    login();
    loadStats();
})