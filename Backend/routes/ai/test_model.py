import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import os

# Load the trained model


while True:
    MODEL_PATH = input("Model-path: ")
    if (MODEL_PATH == ""):
        MODEL_PATH = 'model/ttt_model.h5'

    try:
        model = models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        break
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

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


def ai_move(board, temperature=0.7):
    # 1. Get all legal moves
    available_moves = get_available_moves(board)
    if not available_moves:
        return -1, 0.0

    # 2. Ask the model for move probabilities
    predictions = model.predict(np.array([board]), verbose=0)[0]

    # 3. Keep only probabilities for legal moves
    move_probs = []
    for move in available_moves:
        move_probs.append(predictions[move])

    move_probs = np.array(move_probs)

    # 4. Apply temperature (controls randomness)
    # lower = safer, higher = more random
    move_probs = np.log(move_probs + 1e-9) / temperature
    move_probs = np.exp(move_probs)
    move_probs = move_probs / np.sum(move_probs)

    # 5. Randomly choose a move using these probabilities
    chosen_index = np.random.choice(len(available_moves), p=move_probs)

    chosen_move = available_moves[chosen_index]
    confidence = move_probs[chosen_index]

    return chosen_move, float(confidence)

def play_game(ai_first=False):
    """Play a game against the AI"""
    board = [0.0] * 9
    turn = 1 if ai_first else -1  # 1 for AI (O), -1 for Human (X)
    moves_count = 0

    print("\n" + "="*50)
    print("TIC TAC TOE - AI vs HUMAN TEST")
    print("="*50)
    print("You are X (-1), AI is O (1)")
    print("Positions: 0 1 2")
    print("           3 4 5")
    print("           6 7 8")

    while True:
        print_board(board)

        if turn == 1:  # AI's turn
            move, confidence = ai_move(board)
            if move == -1:
                print("Draw!")
                break
            board[move] = 1
            print(f"AI played at position {move} (confidence: {confidence:.4f})")
            moves_count += 1
        else:  # Human's turn
#            print_board(board)
            try:
                move = int(input("Your move (0-8): "))
                if move < 0 or move > 8 or board[move] != 0:
                    print("Invalid move!")
                    continue
                board[move] = -1
                moves_count += 1
            except ValueError:
                print("Invalid input!")
                continue

        # Check winner
        winner = check_winner(board)
        if winner == 1:
            print_board(board)
            print("AI wins!")
            return 1
        elif winner == -1:
            print_board(board)
            print("You win!")
            return -1
        elif winner == "Draw":
            print_board(board)
            print("Draw!")
            return 0

        turn *= -1

if __name__ == "__main__":
    while True:
        print("\n1. Play against AI")
        print("2. Exit")
        choice = input("\nChoice (1-2): ")

        if choice == "1":
            if (input("Who starts (0 for AI; 1 for You): ") == "0"):
                play_game(ai_first=True)
            else:
                play_game(ai_first=False)
            input("\nPress Enter to continue...")
        elif choice == "2":
            print("Goodbye!")
            break
        else:
            print("Invalid choice")