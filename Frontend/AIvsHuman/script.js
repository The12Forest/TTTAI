let id = 0
let xturn = true

for (let Y = 1; Y <= 3; Y++) {
    for (let X = 1; X <= 3; X++) {
        document.getElementById("card_" + X + "-" + Y).addEventListener("click", (e) => {clicked(e)});
        document.getElementById("card_" + X + "-" + Y).dataset.X = X;
        document.getElementById("card_" + X + "-" + Y).dataset.Y = Y;
        document.getElementById("card_" + X + "-" + Y).textContent = X + "-" + Y;
    }
}

function clicked(e) {
    // if (e.target.classList.contains("card")) console.log("clicked");
    // e.target.remove()
    // e.target.style.display = "none";
    let target = e.target
    // if (target.textContent != "") {return;}
    if (xturn) {
        target.textContent = "X";
        xturn = false
    } else {
        target.textContent = "O";
        xturn = true;
    }
    console.log(target.dataset.X + "-" + target.dataset.Y)
    check(target.dataset.X, target.dataset.Y);
}

function check(X, Y) {
    // Determine current player (xturn toggled after move, so check opposite)
    const currentPlayer = xturn ? "O" : "X";
    
    // Directions to check: [deltaX, deltaY]
    const directions = [
        [1, 0],   // horizontal
        [0, 1],   // vertical
        [1, 1],   // diagonal down-right
        [1, -1]   // diagonal up-right
    ];
    
    for (let [dx, dy] of directions) {
        let count = 1; // Count the current piece
        
        // Check in positive direction
        let x = parseInt(X) + dx;
        let y = parseInt(Y) + dy;
        while (x >= 1 && x <= 3 && y >= 1 && y <= 3) {
            let cell = document.getElementById("card_" + x + "-" + y);
            if (cell.textContent === currentPlayer) {
                count++;
            } else {
                break;
            }
            x += dx;
            y += dy;
        }
        
        // Check in negative direction
        x = parseInt(X) - dx;
        y = parseInt(Y) - dy;
        while (x >= 1 && x <= 3 && y >= 1 && y <= 3) {
            let cell = document.getElementById("card_" + x + "-" + y);
            if (cell.textContent === currentPlayer) {
                count++;
            } else {
                break;
            }
            x -= dx;
            y -= dy;
        }
        
        if (count >= 3) {
            console.log(currentPlayer + " wins!");
            return true;
        }
    }
    
    return false;
}

