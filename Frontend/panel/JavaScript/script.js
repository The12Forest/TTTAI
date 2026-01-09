const username = getCookie("username");
document.getElementById("username").innerText = capitalizeFirstLetter(username);
if (!username) {
    window.location.href = "/login";
}