import * as tf from '@tensorflow/tfjs';

const WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
];

function checkWinner(board) {
    for (const [a, b, c] of WIN_LINES) {
        if (board[a] !== 0 && board[a] === board[b] && board[a] === board[c]) {
            return board[a]; // 1 for X, 2 for O
        }
    }
    return null;
}

function hasLegalMoves(board) {
    return board.includes(0);
}

function getBestLegalMove(board, modelOutput, epsilon = 0.0) {
    const legalMoves = board.map((v, i) => v === 0 ? i : -1).filter(i => i !== -1);
    
    if (legalMoves.length === 0) return null;
    
    // Epsilon-greedy
    if (Math.random() < epsilon) {
        return legalMoves[Math.floor(Math.random() * legalMoves.length)];
    }
    
    let bestMove = legalMoves[0];
    let bestScore = modelOutput[bestMove];
    
    for (let i = 1; i < legalMoves.length; i++) {
        if (modelOutput[legalMoves[i]] > bestScore) {
            bestScore = modelOutput[legalMoves[i]];
            bestMove = legalMoves[i];
        }
    }
    
    return bestMove;
}

function playGame(model, epsilon = 0.1) {
    const board = new Array(9).fill(0);
    let moveCount = 0;
    
    while (hasLegalMoves(board)) {
        // AI move
        const boardTensor = tf.tensor2d([board]);
        const modelOutputTensor = model.predict(boardTensor);
        const modelOutput = Array.from(modelOutputTensor.dataSync());
        
        boardTensor.dispose();
        modelOutputTensor.dispose();
        
        const aiMove = getBestLegalMove(board, modelOutput, epsilon);
        if (aiMove === null) break;
        
        board[aiMove] = 1;
        
        if (checkWinner(board) === 1) {
            return true; // AI won
        }
        
        if (!hasLegalMoves(board)) break;
        
        // Opponent move
        const empty = board.map((v, i) => v === 0 ? i : -1).filter(i => i !== -1);
        if (empty.length > 0) {
            board[empty[Math.floor(Math.random() * empty.length)]] = 2;
        }
        
        if (checkWinner(board) === 2) break; // Opponent won
        moveCount++;
    }
    
    return false; // Didn't win
}

async function main() {
    try {
        console.log('Loading model from tictactoe_model.keras...');
        const model = await tf.loadLayersModel('file://./tictactoe_model.keras');
        console.log('✓ Model loaded successfully!');
        
        // Test on empty board
        console.log('\n=== Testing on empty board ===');
        const emptyBoard = new Array(9).fill(0);
        const testTensor = tf.tensor2d([emptyBoard]);
        const prediction = model.predict(testTensor);
        const predictionData = Array.from(prediction.dataSync());
        
        console.log('Prediction scores for each position:');
        predictionData.forEach((score, idx) => {
            console.log(`  Position ${idx}: ${score.toFixed(4)}`);
        });
        
        const bestMoveIdx = predictionData.indexOf(Math.max(...predictionData));
        console.log(`\n✓ Best opening move: position ${bestMoveIdx}`);
        
        testTensor.dispose();
        prediction.dispose();
        
        // Test multiple games
        console.log('\n=== Testing 10 games ===');
        let wins = 0;
        for (let i = 0; i < 10; i++) {
            if (playGame(model, 0)) {
                wins++;
            }
        }
        console.log(`Win rate: ${wins}/10 (${wins * 10}%)`);
        
        model.dispose();
        console.log('\n✓ Model test complete!');
    } catch (error) {
        console.error('Error loading model:', error);
    }
}

main();