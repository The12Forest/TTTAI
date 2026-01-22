"""
===============================================================================
LEARN FROM HUMAN GAMEPLAY - FINE-TUNE AI BY PLAYING AGAINST IT
===============================================================================
This script:
1. Loads an existing H5 model
2. Lets you play against the AI
3. When you WIN, the AI learns from your moves (its mistakes)
4. Fine-tunes the model to play better

Beat the AI to teach it!
===============================================================================
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
import os
import json

# ============================================================================
# GAME LOGIC
# ============================================================================

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
    (0, 4, 8), (2, 4, 6)              # Diagonals
]

def check_winner(board):
    """Check for winner: 1=X wins, -1=O wins, 0=ongoing, 'Draw'=tie"""
    for a, b, c in WIN_LINES:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return 0 if 0 in board else "Draw"

def get_available_moves(board):
    return [i for i in range(9) if board[i] == 0]

def print_board(board):
    """Pretty print the board"""
    symbols = {0: '·', 1: 'X', -1: 'O'}
    print()
    print("    0   1   2")
    print("  ┌───┬───┬───┐")
    for row in range(3):
        cells = [symbols[int(board[row * 3 + col])] for col in range(3)]
        print(f"{row} │ {cells[0]} │ {cells[1]} │ {cells[2]} │")
        if row < 2:
            print("  ├───┼───┼───┤")
    print("  └───┴───┴───┘")
    print()

# ============================================================================
# AI PLAYER
# ============================================================================

def ai_move(model, board):
    """Get AI's move using the neural network"""
    predictions = model.predict(np.array([board], dtype=np.float32), verbose=0)[0]
    
    # Mask illegal moves
    available = get_available_moves(board)
    for i in range(9):
        if i not in available:
            predictions[i] = -999
    
    best_move = np.argmax(predictions)
    confidence = predictions[best_move]
    
    return best_move, confidence

# ============================================================================
# PLAY GAME
# ============================================================================

def play_game(model, human_starts=True):
    """Play a single game. Returns (winner, game_history, human_symbol)"""
    board = [0] * 9
    game_history = []
    
    human_symbol = 1   # Human is X
    ai_symbol = -1     # AI is O
    
    current_player = human_symbol if human_starts else ai_symbol
    
    print("\n" + "=" * 50)
    if human_starts:
        print("NEW GAME - You start! (You are X)")
    else:
        print("NEW GAME - AI starts! (You are X)")
    print("=" * 50)
    print("Positions: 0|1|2 / 3|4|5 / 6|7|8")
    
    while True:
        print_board(board)
        
        result = check_winner(board)
        if result != 0:
            if result == human_symbol:
                print("YOU WIN! AI will learn from this.")
            elif result == ai_symbol:
                print("AI WINS!")
            else:
                print("DRAW!")
            return result if result != "Draw" else 0, game_history, human_symbol
        
        available = get_available_moves(board)
        if not available:
            print("DRAW!")
            return 0, game_history, human_symbol
        
        if current_player == human_symbol:
            print(f"Your turn (X). Available: {available}")
            
            while True:
                try:
                    move_input = input("Enter position (0-8) or 'q' to quit: ").strip()
                    if move_input.lower() == 'q':
                        return None, game_history, human_symbol
                    move = int(move_input)
                    if move not in available:
                        print(f"Invalid! Choose from: {available}")
                        continue
                    break
                except ValueError:
                    print("Enter a number 0-8")
            
            game_history.append((board.copy(), move, human_symbol))
            board[move] = human_symbol
            print(f"You played: {move}")
        else:
            print("AI is thinking... ")
            move, confidence = ai_move(model, board)
            game_history.append((board.copy(), move, ai_symbol))
            board[move] = ai_symbol
            print(f"AI played: {move} (confidence: {confidence:.3f})")
        
        current_player = ai_symbol if current_player == human_symbol else human_symbol

# ============================================================================
# DATA COLLECTION & AUGMENTATION
# ============================================================================

def collect_training_data(game_history, winner, human_symbol):
    """Collect training data ONLY from games the human won"""
    training_samples = []
    
    if winner != human_symbol:
        return training_samples
    
    for board_state, move, player in game_history:
        if player == human_symbol:
            target = [0.0] * 9
            target[move] = 1.0
            training_samples.append([list(board_state), target])
    
    return training_samples

def augment_with_symmetries(board, targets):
    """Generate all 8 symmetries of a board position"""
    def rotate_90(arr):
        return [arr[6], arr[3], arr[0], arr[7], arr[4], arr[1], arr[8], arr[5], arr[2]]
    
    def flip_horizontal(arr):
        return [arr[2], arr[1], arr[0], arr[5], arr[4], arr[3], arr[8], arr[7], arr[6]]
    
    symmetries = []
    curr_board, curr_targets = list(board), list(targets)
    
    for _ in range(4):
        symmetries.append((curr_board.copy(), curr_targets.copy()))
        curr_board = rotate_90(curr_board)
        curr_targets = rotate_90(curr_targets)
    
    curr_board = flip_horizontal(list(board))
    curr_targets = flip_horizontal(list(targets))
    
    for _ in range(4):
        symmetries.append((curr_board.copy(), curr_targets.copy()))
        curr_board = rotate_90(curr_board)
        curr_targets = rotate_90(curr_targets)
    
    return symmetries

# ============================================================================
# FINE-TUNING
# ============================================================================

def finetune_model(model, training_data, epochs=50):
    """Fine-tune the model on collected human gameplay data"""
    print("\n" + "=" * 50)
    print("FINE-TUNING MODEL ON YOUR GAMEPLAY")
    print("=" * 50)
    
    X, y = [], []
    for board, target in training_data:
        for sym_board, sym_target in augment_with_symmetries(board, target):
            X.append(sym_board)
            y.append(sym_target)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    print(f"Training: {len(training_data)} samples → {len(X)} augmented")
    
    model.optimizer.learning_rate.assign(0.00005)
    
    early_stop = callbacks.EarlyStopping(
        monitor='loss', patience=15, restore_best_weights=True, verbose=1
    )
    
    model.fit(X, y, epochs=epochs, batch_size=min(32, len(X)),
              callbacks=[early_stop], verbose=1, shuffle=True)
    
    print("Fine-tuning complete!")
    return model

# ============================================================================
# SAVE MODEL
# ============================================================================

def save_model(model, save_path):
    """Save the model"""
    model.save(save_path)
    print(f"Model saved to: {save_path}")
    
    tfjs_dir = os.path.splitext(save_path)[0] + "_tfjs"
    try:
        os.system(f'tensorflowjs_converter --input_format=keras "{save_path}" "{tfjs_dir}"')
        if os.path.exists(os.path.join(tfjs_dir, 'model.json')):
            print(f"TensorFlow.js saved to: {tfjs_dir}")
    except Exception as e:
        print(f"TensorFlow.js conversion skipped: {e}")

def save_training_data(training_data, filepath):
    """Save training data to JSON"""
    with open(filepath, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"Data saved to: {filepath}")

def load_training_data(filepath):
    """Load training data from JSON"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return []

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LEARN FROM HUMAN GAMEPLAY")
    print("=" * 60)
    print("\nBeat the AI to teach it your moves!")
    print("The AI learns from its MISTAKES (when you win).\n")
    
    # Step 1: Get model path
    while True:
        model_path = input("Enter path to H5 model (or Enter for default): ").strip()
        
        if not model_path:
            default_paths = [
                'output_model/ttt_model_perfect.h5',
                'ttt_model_smart.h5',
                'ttt_model_perfect.h5',
                'finetuned_model/ttt_model_finetuned.h5'
            ]
            for path in default_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"Using: {model_path}")
                    break
            else:
                print("No default model found.")
                continue
        
        if not os.path.exists(model_path):
            print(f"File not found: {model_path}")
            continue
        break
    
    # Step 2: Load model
    print(f"\nLoading model...")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded!")
    except Exception as e:
        print(f"Failed: {e}")
        exit(1)
    
    # Step 3: Initialize
    training_data = []
    data_file = "human_gameplay_data.json"
    
    if os.path.exists(data_file):
        existing = load_training_data(data_file)
        if existing:
            print(f"Found {len(existing)} existing samples")
            if input("Load them? (y/n): ").strip().lower() == 'y':
                training_data = existing
    
    games_played = 0
    human_wins = 0
    ai_wins = 0
    draws = 0
    human_starts_next = True
    
    print("\n" + "=" * 60)
    print("LET'S PLAY!")
    print("=" * 60)
    
    while True:
        result, history, human_symbol = play_game(model, human_starts=human_starts_next)
        
        if result is None:
            break
        
        games_played += 1
        
        if result == human_symbol:
            human_wins += 1
            new_samples = collect_training_data(history, result, human_symbol)
            training_data.extend(new_samples)
            print(f"Added {len(new_samples)} samples (total: {len(training_data)})")
        elif result == -human_symbol:
            ai_wins += 1
        else:
            draws += 1
        
        print(f"\nStats: {human_wins}W - {ai_wins}L - {draws}D | Samples: {len(training_data)}")
        
        human_starts_next = not human_starts_next
        
        # Offer to fine-tune after 10+ samples
        if len(training_data) >= 10:
            if input("\nFine-tune now? (y/n): ").strip().lower() == 'y':
                save_training_data(training_data, data_file)
                model = finetune_model(model, training_data, epochs=50)
                save_model(model, model_path)
                training_data = []
                
                if input("Continue playing? (y/n): ").strip().lower() != 'y':
                    break
        else:
            if input("\nPlay again? (y/n): ").strip().lower() != 'y':
                break
    
    # Final save
    if training_data:
        print(f"\nSaving {len(training_data)} samples...")
        save_training_data(training_data, data_file)
        
        if input("Fine-tune before exit? (y/n): ").strip().lower() == 'y':
            model = finetune_model(model, training_data, epochs=50)
            save_model(model, model_path)
    
    print("\n" + "=" * 60)
    print(f"Final: {human_wins}W - {ai_wins}L - {draws}D ({games_played} games)")
    print("=" * 60)
