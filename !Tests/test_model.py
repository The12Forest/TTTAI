import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import os

# Load the trained model
MODEL_PATH = 'model/ttt_model.h5'

try:
    model = models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def print_board(board):
    """Print the board in a readable format"""
    symbols = {0: ' ', 1: 'O', -1: 'X'}
    print("\n")
    for i in range(3):
        print(f" {symbols[board[i*3]]} | {symbols[board[i*3+1]]} | {symbols[board[i*3+2]]} ")
        if i < 2:
            print("-----------")
    print("\n")

def check_winner(board):
    """Check if there's a winner"""
    win_states = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
    for line in win_states:
        if board[line[0]] == board[line[1]] == board[line[2]] != 0:
            return board[line[0]]
    return 0 if 0 in board else "Draw"

def get_available_moves(board):
    """Get list of available moves"""
    return [i for i in range(9) if board[i] == 0]

# --- PERFECT MINIMAX WITH ALPHA-BETA PRUNING ---
def minimax(board, is_maximizing, alpha=-float('inf'), beta=float('inf')):
    """Minimax algorithm with alpha-beta pruning"""
    winner = check_winner(board)
    
    # Terminal states
    if winner == 1:  # AI wins
        return 10
    elif winner == -1:  # Human wins
        return -10
    elif winner == "Draw":
        return 0
    
    if is_maximizing:  # AI's turn (trying to maximize score)
        max_eval = -float('inf')
        for move in get_available_moves(board):
            board[move] = 1
            eval_score = minimax(board, False, alpha, beta)
            board[move] = 0
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:  # Human's turn (trying to minimize score)
        min_eval = float('inf')
        for move in get_available_moves(board):
            board[move] = -1
            eval_score = minimax(board, True, alpha, beta)
            board[move] = 0
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval

def perfect_ai_move(board):
    """Get perfect AI move using minimax"""
    best_score = -float('inf')
    best_move = -1
    
    for move in get_available_moves(board):
        board[move] = 1
        score = minimax(board, False)
        board[move] = 0
        
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move, float(best_score)

def neural_ai_move(board):
    """Get AI's move using the neural network model"""
    available_moves = get_available_moves(board)
    
    if not available_moves:
        return -1, 0
    
    # Get predictions
    predictions = model.predict(np.array([board]), verbose=0)[0]
    
    # Filter out illegal moves
    valid_predictions = [(predictions[i], i) for i in available_moves]
    
    # Get best move
    best_value, best_move = max(valid_predictions, key=lambda x: x[0])
    
    return best_move, float(best_value)

def play_game(ai_type='perfect', ai_first=False):
    """Play a game against the AI"""
    board = [0.0] * 9
    turn = 1 if ai_first else -1  # 1 for AI (O), -1 for Human (X)
    moves_count = 0
    
    print("\n" + "="*50)
    print(f"TIC TAC TOE - {ai_type.upper()} AI vs HUMAN TEST")
    print("="*50)
    print("You are X (-1), AI is O (1)")
    print("Positions: 0 1 2")
    print("           3 4 5")
    print("           6 7 8")
    
    while True:
        print_board(board)
        
        if turn == 1:  # AI's turn
            if ai_type == 'perfect':
                move, confidence = perfect_ai_move(board)
                confidence_text = f"score: {confidence}"
            else:
                move, confidence = neural_ai_move(board)
                confidence_text = f"confidence: {confidence:.4f}"
            
            if move == -1:
                print("Draw!")
                break
            board[move] = 1
            print(f"AI played at position {move} ({confidence_text})")
            moves_count += 1
        else:  # Human's turn
            print_board(board)
            try:
                move = int(input("Your move (0-8): "))
                if move < 0 or move > 8 or board[move] != 0:
                    print("❌ Invalid move!")
                    continue
                board[move] = -1
                moves_count += 1
            except ValueError:
                print("❌ Invalid input!")
                continue
        
        # Check winner
        winner = check_winner(board)
        if winner == 1:
            print_board(board)
            print("🤖 AI wins!")
            return 1
        elif winner == -1:
            print_board(board)
            print("👤 You win!")
            return -1
        elif winner == "Draw":
            print_board(board)
            print("🤝 Draw!")
            return 0
        
        turn *= -1

def evaluate_ai(ai_type='perfect', num_games=10):
    """Evaluate AI performance in random games"""
    print("\n" + "="*50)
    print(f"EVALUATING {ai_type.upper()} AI ({num_games} games)")
    print("="*50)
    
    results = {'ai_wins': 0, 'human_wins': 0, 'draws': 0}
    
    for game_num in range(num_games):
        board = [0.0] * 9
        turn = 1  # AI always goes first for consistency
        
        while True:
            available = get_available_moves(board)
            if not available:
                results['draws'] += 1
                break
            
            if turn == 1:  # AI's turn
                if ai_type == 'perfect':
                    move, _ = perfect_ai_move(board)
                else:
                    move, _ = neural_ai_move(board)
                board[move] = 1
            else:  # Human plays random
                import random
                move = random.choice(available)
                board[move] = -1
            
            winner = check_winner(board)
            if winner == 1:
                results['ai_wins'] += 1
                break
            elif winner == -1:
                results['human_wins'] += 1
                break
            elif winner == "Draw":
                results['draws'] += 1
                break
            
            turn *= -1
        
        print(f"Game {game_num + 1}: ", end="")
        if winner == 1:
            print("AI wins ✅")
        elif winner == -1:
            print("Human wins ❌")
        else:
            print("Draw 🤝")
    
    print("\n" + "="*50)
    print("RESULTS:")
    print(f"AI Wins:    {results['ai_wins']}/{num_games} ({results['ai_wins']/num_games*100:.1f}%)")
    print(f"Human Wins: {results['human_wins']}/{num_games} ({results['human_wins']/num_games*100:.1f}%)")
    print(f"Draws:      {results['draws']}/{num_games} ({results['draws']/num_games*100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    while True:
        print("\n" + "="*50)
        print("TIC TAC TOE AI TEST ENVIRONMENT")
        print("="*50)
        print("1. Play against PERFECT AI (Minimax)")
        if model:
            print("2. Play against NEURAL NETWORK AI")
            print("3. Evaluate PERFECT AI (10 games)")
            print("4. Evaluate NEURAL NETWORK AI (10 games)")
            print("5. Exit")
        else:
            print("2. Evaluate PERFECT AI (10 games)")
            print("3. Exit")
        
        choice = input("\nChoice: ")
        
        if choice == "1":
            play_game(ai_type='perfect', ai_first=False)
            input("\nPress Enter to continue...")
        elif choice == "2" and model:
            play_game(ai_type='neural', ai_first=False)
            input("\nPress Enter to continue...")
        elif choice == "3" and model:
            num_games = input("Number of games to evaluate (default 10): ")
            try:
                num_games = int(num_games) if num_games else 10
                evaluate_ai(ai_type='perfect', num_games=num_games)
            except ValueError:
                print("❌ Invalid number")
            input("\nPress Enter to continue...")
        elif choice == "4" and model:
            num_games = input("Number of games to evaluate (default 10): ")
            try:
                num_games = int(num_games) if num_games else 10
                evaluate_ai(ai_type='neural', num_games=num_games)
            except ValueError:
                print("❌ Invalid number")
            input("\nPress Enter to continue...")
        elif (choice == "3" and not model) or (choice == "5" and model) or (choice == "2" and not model):
            print("Goodbye!")
            break
        else:
            print("❌ Invalid choice")

