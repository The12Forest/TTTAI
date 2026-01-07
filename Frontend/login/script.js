let baseurl = "http://localhost/api"

document.getElementById("form").addEventListener("submit", (e) => {
    e.preventDefault()
    
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    
    if (!username || !password) {
        alert("Please enter username and password");
        return;
    }
    
    fetch(baseurl + "/user/login/" + username + "/" + password, {})
        .then((response) => response.json())
        .then((json) => {
            if (json.Okay) {
                alert("Login successful!");
                // window.location.href = "/play/ai";
            } else {
                alert("Login failed: " + json.Reason);
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
  