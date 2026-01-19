import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import random
from collections import defaultdict
import os

# --- 1. PERFECT TEACHER WITH MEMOIZATION ---
memo = {}

def check_winner(board):
    win_states = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
    for line in win_states:
        if board[line[0]] == board[line[1]] == board[line[2]] != 0:
            return board[line[0]]
    return 0 if 0 in board else "Draw"

def board_to_tuple(board):
    """Convert board list to tuple for hashing"""
    return tuple(board)

def minimax_memo(board, is_maximizing):
    """Minimax with memoization for faster computation"""
    board_key = board_to_tuple(board)
    
    if is_maximizing:
        key = (board_key, True)
    else:
        key = (board_key, False)
    
    if key in memo:
        return memo[key]
    
    res = check_winner(board)
    if res == 1: 
        memo[key] = 10
        return 10
    if res == -1: 
        memo[key] = -10
        return -10
    if res == "Draw": 
        memo[key] = 0
        return 0

    scores = []
    for i in range(9):
        if board[i] == 0:
            board[i] = 1 if is_maximizing else -1
            scores.append(minimax_memo(board, not is_maximizing))
            board[i] = 0
    
    result = max(scores) if is_maximizing else min(scores)
    memo[key] = result
    return result

def get_move_scores(board):
    """Get minimax scores for all available moves"""
    scores = [None] * 9
    available = [i for i in range(9) if board[i] == 0]
    
    for move in available:
        board[move] = 1  # AI is always maximizing
        score = minimax_memo(board, False)
        board[move] = 0
        scores[move] = score
    
    return scores

# --- 2. SMART DATA GENERATION ---
print("🧠 Generating 100,000 perfect samples with Minimax...")
X, y_moves, y_scores = [], [], []

for iteration in range(100):
    print(f"Iteration {iteration + 1}/100 - Samples: {len(X)}")
    
    for sample in range(1000):
        board = [0] * 9
        
        # Create random but valid board state
        for step in range(random.randint(0, 6)):
            empty = [j for j, v in enumerate(board) if v == 0]
            if not empty or check_winner(board) != 0:
                break
            board[random.choice(empty)] = 1 if step % 2 == 0 else -1
        
        # Only use non-terminal states
        if check_winner(board) == 0 and 0 in board:
            move_scores = get_move_scores(board)
            
            # Find best move and create training example
            available = [i for i in range(9) if board[i] == 0]
            if available:
                best_move = max(available, key=lambda x: move_scores[x])
                
                # Normalize scores to [0, 1] range
                max_score = max(move_scores[i] for i in available if move_scores[i] is not None)
                min_score = min(move_scores[i] for i in available if move_scores[i] is not None)
                
                normalized_scores = np.zeros(9)
                for i in available:
                    if max_score == min_score:
                        normalized_scores[i] = 0.5
                    else:
                        normalized_scores[i] = (move_scores[i] - min_score) / (max_score - min_score)
                
                X.append(board.copy())
                y_moves.append(best_move)
                y_scores.append(normalized_scores)

X = np.array(X, dtype=np.float32)
y_moves = np.array(y_moves, dtype=np.int32)
y_scores = np.array(y_scores, dtype=np.float32)

print(f"\n✅ Generated {len(X)} samples")
print(f"Dataset shape: {X.shape}")

# --- 3. SMART NEURAL NETWORK ARCHITECTURE ---
# Simplified model without BatchNormalization for TensorFlow.js compatibility
model = models.Sequential([
    layers.Input(shape=(9,)),
    
    # First layer
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    
    # Second layer
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.3),
    
    # Third layer
    layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.2),
    
    # Output layer - predict move quality for each position
    layers.Dense(9, activation='sigmoid')  # Sigmoid for score prediction
])

# Compile with custom learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='mse',  # Mean squared error for continuous score prediction
    metrics=['mae']
)

print("\nModel Architecture:")
model.summary()

# --- 4. SMART TRAINING WITH CALLBACKS ---
print("\n🚀 Starting Smart Training...")

# Early stopping if validation loss doesn't improve
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate if stuck
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

# Train on move quality prediction (y_scores)
history = model.fit(
    X, y_scores,
    epochs=300,
    batch_size=64,
    validation_split=0.15,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
    shuffle=True
)

# --- 5. EVALUATE MODEL QUALITY ---
print("\n📊 Evaluating Model Quality...")
val_predictions = model.predict(X[:5000])
print(f"Model evaluation on training data:")
print(f"Mean Absolute Error: {np.mean(np.abs(val_predictions - y_scores[:5000])):.4f}")

# --- 6. SAVE MODEL ---
print("\n💾 Saving model...")
model.save('ttt_model_smart.h5')
print("✅ Model saved as ttt_model_smart.h5")

# --- 7. CONVERT TO TENSORFLOWJS ---
print("\n🔄 Converting to TensorFlow.js...")
output_dir = 'ttt_model_tfjs'
if os.path.exists(output_dir):
    import shutil
    shutil.rmtree(output_dir)

os.system(f'tensorflowjs_converter --input_format=keras ttt_model_smart.h5 {output_dir}')

if os.path.exists(output_dir):
    print(f"✅ Model converted to TensorFlow.js!")
    print(f"Files created in {output_dir}:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")
    
    # Rebuild model.json for TensorFlow.js browser loading
    import json
    print("\n🔧 Rebuilding model.json for TensorFlow.js...")
    
    model_json_path = os.path.join(output_dir, 'model.json')
    with open(model_json_path, 'r') as f:
        original_data = json.load(f)
    
    # Create clean TensorFlow.js v2 format
    fixed_model = {
        "format": "layers-model",
        "generatedBy": "TensorFlow.js Converter",
        "convertedBy": "Custom Fix",
        "modelTopology": {
            "class_name": "Sequential",
            "config": [
                {
                    "class_name": "InputLayer",
                    "config": {
                        "batch_input_shape": [None, 9],
                        "dtype": "float32",
                        "name": "input_1"
                    }
                },
                {
                    "class_name": "Dense",
                    "config": {
                        "name": "dense",
                        "trainable": True,
                        "dtype": "float32",
                        "units": 256,
                        "activation": "relu",
                        "use_bias": True
                    }
                },
                {
                    "class_name": "Dense",
                    "config": {
                        "name": "dense_1",
                        "trainable": True,
                        "dtype": "float32",
                        "units": 256,
                        "activation": "relu",
                        "use_bias": True
                    }
                },
                {
                    "class_name": "Dense",
                    "config": {
                        "name": "dense_2",
                        "trainable": True,
                        "dtype": "float32",
                        "units": 128,
                        "activation": "relu",
                        "use_bias": True
                    }
                },
                {
                    "class_name": "Dense",
                    "config": {
                        "name": "dense_3",
                        "trainable": True,
                        "dtype": "float32",
                        "units": 9,
                        "activation": "sigmoid",
                        "use_bias": True
                    }
                }
            ]
        },
        "weightsManifest": original_data.get("weightsManifest", [{
            "paths": ["group1-shard1of1.bin"],
            "weights": []
        }])
    }
    
    with open(model_json_path, 'w') as f:
        json.dump(fixed_model, f, indent=2)
    
    print("✅ model.json rebuilt for TensorFlow.js!")
    print(f"Model ready at: {os.path.abspath(output_dir)}")

# --- 8. SAVE FOR BACKEND SERVER ---
print("\n💾 Saving model for backend server...")

# Save in SavedModel format (best for TensorFlow Serving)
saved_model_dir = 'ttt_model_savedmodel'
if os.path.exists(saved_model_dir):
    import shutil
    shutil.rmtree(saved_model_dir)

model.save(saved_model_dir, save_format='tf')
print(f"✅ SavedModel format saved to: {os.path.abspath(saved_model_dir)}")
print(f"   Use this for TensorFlow Serving or Flask/FastAPI backend")

# Also keep the H5 format for quick loading
print(f"✅ H5 format already saved to: {os.path.abspath('ttt_model_smart.h5')}")
print(f"   Use this for quick Python backend loading\n")

def play():
    board = [0.0] * 9
    print("\n🎮 Game On! You are X (-1), AI is O (1).")
    
    while check_winner(board) == 0:
        # AI Turn
        predictions = model.predict(np.array([board]), verbose=0)[0]
        
        # Filter out illegal moves
        for i in range(9):
            if board[i] != 0:
                predictions[i] = -1
        
        move = np.argmax(predictions)
        board[move] = 1
        print(f"AI chose {move} (confidence: {predictions[move]:.4f}). Board: {board}")
        
        if check_winner(board) != 0:
            break
        
        # Human Turn
        try:
            human = int(input("Your move (0-8): "))
            if human < 0 or human > 8 or board[human] != 0:
                print("Invalid move!")
                continue
            board[human] = -1
        except ValueError:
            print("Invalid input!")
            continue

    result = check_winner(board)
    if result == 1:
        print("🤖 AI wins!")
    elif result == -1:
        print("👤 You win!")
    else:
        print("🤝 Draw!")

play()
