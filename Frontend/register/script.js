let baseurl = "http://localhost/api"

document.getElementById("form").addEventListener("submit", (e) => {
    e.preventDefault()
    
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    
    if (!username || !password) {
        alert("Please enter username and password");
        return;
    }
    
    fetch(baseurl + "/user/register/" + username + "/" + password)
        .then((response) => response.json())
        .then((json) => {
            if (json.Okay) {
                alert("Register successful!");
                // window.location.href = "/play/ai";
            } else {
                alert("Register failed: " + json.Reason);
            }
        })
        .catch((error) => {
            console.error("Error:", error);
            alert("An error occurred during login");
        });
})

window.addEventListener("DOMContentLoaded", () => {
    document.getElementById("username").focus();
});
  