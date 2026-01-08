let baseurl = "https://localhost/api"

async function sha256(message) {
    const msgBuffer = new TextEncoder().encode(message);
    const hashBuffer = await crypto.subtle.digest("SHA-256", msgBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, "0")).join("");
    return hashHex;
}

document.getElementById("form").addEventListener("submit", async (e) => {
    e.preventDefault()
    
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    
    if (!username || !password) {
        alert("Please enter username and password");
        return;   
    }

    let hashPassword = await sha256(username + "||" + password)
    
    fetch(baseurl + "/user/register/" + username + "/" + hashPassword)
        .then((response) => response.json())
        .then((json) => {
            if (json.Okay) {
                alert("Register successful!");
                
                let expires
                if (document.getElementById("ssign").checked) {
                    let date = new Date();
                    date.setTime(date.getTime() + (365 * 24 * 60 * 60 * 1000)); // 365 days in milliseconds
                    expires = "expires=" + date.toUTCString();
                } else {
                    let date = new Date();
                    date.setTime(date.getTime() + (1 * 24 * 60 * 60 * 1000)); // 1 days in milliseconds
                    expires = "expires=" + date.toUTCString();
                }
                document.cookie = "username=" + username + "; expires=" + expires + ";path=/;"; 
                document.cookie = "password=" + hashPassword + "; expires=" + expires + ";path=/;"; 


                window.location.href = "/panel";
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
  
