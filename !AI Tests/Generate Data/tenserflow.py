import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import random
import os

# Win lines for tic-tac-toe
WIN_LINES = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    [0, 4, 8], [2, 4, 6]
]

MODEL_PATH = 'tictactoe_model.keras'

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
    """Build improved neural network model"""
    model = Sequential([
        Dense(128, activation='relu', input_dim=9),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(9, activation='sigmoid')
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

def train_model(model, X_train, Y_train, epochs=50):
    """Train model with early stopping"""
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    
    model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=16,
        verbose=0,
        callbacks=[early_stop]
    )
    return model

def generate_training_data(games_count=100):
    """Generate diverse training data from random games"""
    X_data = []
    Y_data = []
    
    for _ in range(games_count):
        board = [0] * 9
        move_count = 0
        
        while has_legal_moves(board):
            # Record state before AI move
            X_data.append(board.copy())
            
            # AI picks best legal move (random for diversity in data gen)
            legal_moves = [i for i in range(9) if board[i] == 0]
            ai_move = random.choice(legal_moves)
            
            # Create one-hot output
            move_one_hot = [0] * 9
            move_one_hot[ai_move] = 1
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
            if move_count > 10:  # Safety limit
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
X_data, Y_data = generate_training_data(200)

print("Building model...")
model = load_or_create_model()

while True:
    new_X = []
    new_Y = []
    wins = 0

    for i in range(300):
        states, moves, won = play_game(model, epsilon=0.5)
        if len(states) > 0:
            new_X.append(states)
            new_Y.append(moves)
            if won:
                wins += 1
        if (i + 1) % 10 == 0:
            print(f"  Played {i + 1} games, wins: {wins}")
        
        if (i > 5):
            print(f"Win rate: {wins}/{i+1} ({100*wins/(i+1):.1f}%)")
            if (wins / (i+1)) < 0.6:
                break

    
    if (wins/300 > 0.6):
        break

# Combine data
if new_X:
    X_new = np.vstack(new_X)
    Y_new = np.vstack(new_Y)
    X_data = np.vstack(X_new)
    Y_data = np.vstack(Y_new)
    print(f"Dataset size: {len(X_data)}")












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



 