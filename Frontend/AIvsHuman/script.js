let id = 0
let xturn = true
let round = 0
for (let Y = 1; Y <= 3; Y++) {
    for (let X = 1; X <= 3; X++) {
        document.getElementById("card_" + X + "-" + Y).addEventListener("click", (e) => {clicked(e)});
        document.getElementById("card_" + X + "-" + Y).dataset.X = X;
        document.getElementById("card_" + X + "-" + Y).dataset.Y = Y;
        // document.getElementById("card_" + X + "-" + Y).classList.add("active");

        // document.getElementById("card_" + X + "-" + Y).textContent = X + "-" + Y;
    }
}

function clicked(e) {
    let target = e.target
    if (target.textContent != "") {return;}
    if (xturn) {
        target.textContent = "X";
        xturn = false
        round++
    } else {
        target.textContent = "O";
        xturn = true;
        round++;
    }
    
    console.log(target.dataset.X + "-" + target.dataset.Y)
    let winner = check(target.dataset.X, target.dataset.Y);
    if (winner.wins) {
        banner("asdf");
        document.getElementById("banner").textContent = "Winner is: " + target.textContent
    }

    
}

function check(X, Y) {
    
    const currentPlayer = xturn ? "O" : "X";
    
    // [deltaX, deltaY]
    const directions = [
        [1, 0],
        [0, 1],
        [1, 1],
        [1, -1]
    ];
    
    for (let [dx, dy] of directions) {
        let count = 1; 
        
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
            return {"wins": true, "Player": currentPlayer};
        }
    }
    
    return {"wins": false, "Player": currentPlayer};
}

function banner(winner) {
    document.getElementById("banner").classList.toggle("show");
}

