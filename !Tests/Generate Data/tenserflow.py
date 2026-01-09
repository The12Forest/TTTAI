import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import random
import os

# Win lines for tic-tac-toe
WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
]

MODEL_PATH = 'tictactoe_model.keras'
BATCH_SIZE = 32
MAX_DATASET_SIZE = 10000

def check_winner(board):
    """Check if there's a winner. 1=X, 2=O, None=no winner"""
    for a, b, c in WIN_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return None

def is_board_full(board):
    """Check if board is full"""
    return 0 not in board

def has_legal_moves(board):
    """Check if there are legal moves"""
    return 0 in board

def get_board_symmetries(board):
    """Generate all 8 symmetries of a board (rotations + reflections)"""
    board_arr = np.array(board).reshape(3, 3)
    symmetries = [board]
    
    # Rotations
    for _ in range(3):
        board_arr = np.rot90(board_arr)
        symmetries.append(board_arr.flatten().tolist())
    
    # Reflections
    board_arr = np.array(board).reshape(3, 3)
    for _ in range(3):
        board_arr = np.rot90(np.fliplr(board_arr))
        symmetries.append(board_arr.flatten().tolist())
    
    return symmetries





def get_best_legal_move(board, model_output, epsilon=0.0):
    """Get best legal move with optional epsilon-greedy exploration"""
    legal_moves = [i for i in range(9) if board[i] == 0]
    
    if not legal_moves:
        return None
    
    # Epsilon-greedy strategy
    if random.random() < epsilon:
        return random.choice(legal_moves)
    
    # Find best move from legal moves
    best_move = legal_moves[0]
    best_score = model_output[legal_moves[0]]
    
    for move in legal_moves[1:]:
        if model_output[move] > best_score:
            best_score = model_output[move]
            best_move = move
    
    return best_move

def build_model():
    """Build improved neural network model with better architecture"""
    model = Sequential([
        Dense(256, activation='relu', input_dim=9),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.15),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(9, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

def train_model(model, X_train, Y_train, epochs=50, validation_split=0.1):
    """Train model with early stopping and learning rate reduction"""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, min_delta=0.001),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
    ]
    
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_split=validation_split,
        verbose=1,
        callbacks=callbacks
    )
    return model, history

def generate_training_data(games_count=200):
    """Generate diverse training data with data augmentation"""
    X_data = []
    Y_data = []
    
    for game_idx in range(games_count):
        board = [0] * 9
        move_count = 0
        
        while has_legal_moves(board):
            # Record state before AI move
            legal_moves = [i for i in range(9) if board[i] == 0]
            ai_move = random.choice(legal_moves)
            
            # Create target with bonus for center and corners (strategic positions)
            move_one_hot = [0] * 9
            move_one_hot[ai_move] = 1
            
            X_data.append(board.copy())
            Y_data.append(move_one_hot)
            
            # Apply AI move
            board[ai_move] = 1
            
            if check_winner(board):
                break
            if not has_legal_moves(board):
                break
            
            # Random opponent move
            empty = [i for i in range(9) if board[i] == 0]
            if empty:
                board[random.choice(empty)] = 2
            
            if check_winner(board):
                break
            
            move_count += 1
            if move_count > 5:
                break
        
        # Data augmentation with board symmetries
        if game_idx % 100 == 0 and len(X_data) > 0:
            recent_boards = X_data[-min(50, len(X_data)):]
            for board in recent_boards:
                for sym_board in get_board_symmetries(board):
                    # Find corresponding move in symmetry
                    for i, orig_board in enumerate(X_data[-len(recent_boards):]):
                        if orig_board == board:
                            X_data.append(sym_board)
                            Y_data.append(Y_data[-(len(recent_boards) - i)])
                            break
    
    return np.array(X_data), np.array(Y_data)

def play_game(model, epsilon=0.1):
    """Play a game and return winning moves"""
    board = [0] * 9
    game_states = []
    game_moves = []
    
    while has_legal_moves(board):
        # AI move
        game_states.append(board.copy())
        
        model_output = model.predict(np.array([board]), verbose=0)[0]
        ai_move = get_best_legal_move(board, model_output, epsilon=epsilon)
        
        if ai_move is None:
            break
        
        move_one_hot = [0] * 9
        move_one_hot[ai_move] = 1
        game_moves.append(move_one_hot)
        
        board[ai_move] = 1
        
        if check_winner(board) == 1:  # AI wins
            return np.array(game_states), np.array(game_moves), True
        
        if not has_legal_moves(board):
            break
        
        # Random opponent
        empty = [i for i in range(9) if board[i] == 0]
        if empty:
            board[random.choice(empty)] = 2
        
        if check_winner(board) == 2:  # Opponent wins
            break
    
    return np.array(game_states), np.array(game_moves), False

def load_or_create_model():
    """Load existing model or create new one"""
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}")
        return load_model(MODEL_PATH)
    return build_model()

# ============== TRAINING LOOP ==============
print("Generating initial training data...")
X_data, Y_data = generate_training_data(300)
print(f"Generated {len(X_data)} initial training samples")

print("Building/Loading model...")
model = load_or_create_model()

# Warm-up training on initial data
print("\n=== Warm-up Phase ===")
model, _ = train_model(model, X_data, Y_data, epochs=40)
model.save(MODEL_PATH)

# Self-play training loop
num_phases = 200
games_per_phase = 100

for phase in range(num_phases):
    print(f"\n=== Phase {phase + 1}/{num_phases} ===")
    
    # Progressive difficulty
    epsilon = max(0.01, 0.2 * (1 - phase / num_phases))
    
    print(f"Playing {games_per_phase} games (epsilon={epsilon:.3f})...")
    new_X = []
    new_Y = []
    wins = 0
    draws = 0
    
    for i in range(games_per_phase):
        states, moves, won = play_game(model, epsilon=epsilon)
        if len(states) > 0:
            new_X.append(states)
            new_Y.append(moves)
            if won:
                wins += 1
            else:
                draws += 1
        
        if (i + 1) % 20 == 0:
            print(f"  Played {i + 1} games, wins: {wins}, draws: {draws}")
    
    win_rate = wins / games_per_phase if games_per_phase > 0 else 0
    print(f"Win rate: {wins}/{games_per_phase} ({100*win_rate:.1f}%)")
    
    # Combine data
    if new_X:
        X_new = np.vstack(new_X)
        Y_new = np.vstack(new_Y)
        X_data = np.vstack([X_data, X_new])
        Y_data = np.vstack([Y_data, Y_new])
        
        # Keep dataset size bounded
        if len(X_data) > MAX_DATASET_SIZE:
            # Keep most recent data
            X_data = X_data[-MAX_DATASET_SIZE:]
            Y_data = Y_data[-MAX_DATASET_SIZE:]
        
        print(f"Dataset size: {len(X_data)}")
        
        # Train on accumulated data
        print(f"Training on {len(X_data)} samples...")
        model, history = train_model(model, X_data, Y_data, epochs=25, validation_split=0.15)
        
        # Save model
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    
    # Stop early if converged
    if win_rate > 0.85 and phase > 50:
        print("\nModel converged! Continuing for refinement...")












# Training phases
num_phases = 300
games_per_phase = 60

for phase in range(num_phases):
    print(f"\n=== Phase {phase + 1}/{num_phases} ===")
    if (num_phases == 100):
        games_per_phase += games_per_phase
    if (num_phases == 1000):
        games_per_phase += games_per_phase*2

    # Train on existing data
    print(f"Training on {len(X_data)} samples...")
    model = train_model(model, X_data, Y_data, epochs=30)
    
    # Play games to generate new data
    print(f"Playing {games_per_phase} games...")
    new_X = []
    new_Y = []
    wins = 0
    
    epsilon = 0.05 * (1 - phase / num_phases)  # Decrease exploration
    
    for i in range(games_per_phase):
        states, moves, won = play_game(model, epsilon=epsilon)
        if len(states) > 0:
            new_X.append(states)
            new_Y.append(moves)
            if won:
                wins += 1
        if (i + 1) % 10 == 0:
            print(f"  Played {i + 1} games, wins: {wins}")
    
    print(f"Win rate: {wins}/{games_per_phase} ({100*wins/games_per_phase:.1f}%)")
    
    # Combine data
    if (wins/games_per_phase > 0.6):
        if new_X:
            X_new = np.vstack(new_X)
            Y_new = np.vstack(new_Y)
            X_data = np.vstack([X_data, X_new])
            Y_data = np.vstack([Y_data, Y_new])
            print(f"Dataset size: {len(X_data)}")
    
    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

print("\n=== Training Complete ===")
print(f"Final model saved to {MODEL_PATH}")



 