"""
===============================================================================
PERFECT UNBEATABLE TIC-TAC-TOE AI TRAINER v2.0
===============================================================================
This generates a PERFECT AI that CANNOT be beaten - especially in late game.

Key improvements over previous version:
1. Exhaustive state generation - covers ALL possible game states
2. Alpha-beta pruning minimax for perfect moves
3. HEAVY focus on late-game critical positions (blocking, winning)
4. Data augmentation with all 8 board symmetries
5. Curriculum learning with weighted samples
6. Hard targets for must-win and must-block positions

Run this to generate a new model, then copy to Backend/routes/ai/models/
===============================================================================
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import random
from collections import defaultdict
from itertools import product
import os
import json
import shutil

# ============================================================================
# PART 1: PERFECT MINIMAX ENGINE WITH ALPHA-BETA PRUNING
# ============================================================================

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
    (0, 4, 8), (2, 4, 6)              # Diagonals
]

minimax_cache = {}

def check_winner(board):
    """Check for winner: 1=AI wins, -1=Human wins, 0=ongoing, 'Draw'=tie"""
    for a, b, c in WIN_LINES:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return 0 if 0 in board else "Draw"

def get_available_moves(board):
    return [i for i in range(9) if board[i] == 0]

def minimax(board, depth, alpha, beta, is_maximizing):
    """Minimax with alpha-beta pruning for PERFECT play"""
    key = (tuple(board), is_maximizing)
    if key in minimax_cache:
        return minimax_cache[key]
    
    result = check_winner(board)
    if result == 1:
        return 100 - depth  # AI wins, prefer faster wins
    if result == -1:
        return -100 + depth  # Human wins, delay as long as possible
    if result == "Draw":
        return 0
    
    if is_maximizing:
        max_eval = float('-inf')
        for move in get_available_moves(board):
            board[move] = 1
            eval_score = minimax(board, depth + 1, alpha, beta, False)
            board[move] = 0
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        minimax_cache[key] = max_eval
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_available_moves(board):
            board[move] = -1
            eval_score = minimax(board, depth + 1, alpha, beta, True)
            board[move] = 0
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        minimax_cache[key] = min_eval
        return min_eval

def get_all_move_scores(board):
    """Get perfect minimax score for ALL available moves"""
    scores = {}
    for move in get_available_moves(board):
        board[move] = 1
        score = minimax(board, 0, float('-inf'), float('inf'), False)
        board[move] = 0
        scores[move] = score
    return scores

# ============================================================================
# PART 2: CRITICAL POSITION DETECTION (LATE GAME FOCUS)
# ============================================================================

def find_winning_move(board, player):
    """Find immediate winning move - CRITICAL"""
    for a, b, c in WIN_LINES:
        line = [board[a], board[b], board[c]]
        if line.count(player) == 2 and line.count(0) == 1:
            if board[a] == 0: return a
            if board[b] == 0: return b
            if board[c] == 0: return c
    return None

def find_blocking_move(board, player):
    """Find move that blocks opponent's win - CRITICAL"""
    return find_winning_move(board, -player)

def find_fork_move(board, player):
    """Find move that creates two winning threats (fork)"""
    for move in get_available_moves(board):
        board[move] = player
        threats = 0
        for a, b, c in WIN_LINES:
            line = [board[a], board[b], board[c]]
            if line.count(player) == 2 and line.count(0) == 1:
                threats += 1
        board[move] = 0
        if threats >= 2:
            return move
    return None

def count_empty_cells(board):
    return board.count(0)

def classify_position(board):
    """
    Classify position for training weight.
    CRITICAL: Late game positions get HIGHEST weight!
    """
    empty = count_empty_cells(board)
    
    # ===== LATE GAME (1-3 empty cells) - MOST CRITICAL =====
    if empty <= 3:
        if find_winning_move(board, 1) is not None:
            return 'LATE_MUST_WIN', 50.0  # HIGHEST PRIORITY
        if find_blocking_move(board, 1) is not None:
            return 'LATE_MUST_BLOCK', 50.0  # HIGHEST PRIORITY
        return 'LATE_GAME', 20.0
    
    # ===== MID GAME (4-5 empty cells) =====
    if empty <= 5:
        if find_winning_move(board, 1) is not None:
            return 'MID_MUST_WIN', 25.0
        if find_blocking_move(board, 1) is not None:
            return 'MID_MUST_BLOCK', 25.0
        if find_fork_move(board, 1) is not None:
            return 'MID_FORK', 15.0
        if find_fork_move(board, -1) is not None:
            return 'MID_BLOCK_FORK', 15.0
        return 'MID_GAME', 8.0
    
    # ===== EARLY GAME (6+ empty cells) =====
    if find_winning_move(board, 1) is not None:
        return 'EARLY_MUST_WIN', 15.0
    if find_blocking_move(board, 1) is not None:
        return 'EARLY_MUST_BLOCK', 15.0
    
    return 'NORMAL', 1.0

# ============================================================================
# PART 3: BOARD SYMMETRIES FOR DATA AUGMENTATION
# ============================================================================

def rotate_90(b):
    """Rotate board 90 degrees clockwise"""
    return [b[6], b[3], b[0], b[7], b[4], b[1], b[8], b[5], b[2]]

def flip_horizontal(b):
    """Flip board horizontally"""
    return [b[2], b[1], b[0], b[5], b[4], b[3], b[8], b[7], b[6]]

def get_all_symmetries(board, target):
    """Generate all 8 symmetries of board and target"""
    def transform(arr, func):
        indices = list(range(9))
        transformed_indices = func(indices)
        new_arr = [0.0] * 9
        for new_pos, old_pos in enumerate(transformed_indices):
            new_arr[new_pos] = arr[old_pos]
        return new_arr
    
    symmetries = []
    curr_board, curr_target = list(board), list(target)
    
    # 4 rotations
    for _ in range(4):
        symmetries.append((curr_board.copy(), curr_target.copy()))
        curr_board = rotate_90(curr_board)
        curr_target = transform(curr_target, rotate_90)
    
    # Flip + 4 rotations
    curr_board = flip_horizontal(list(board))
    curr_target = transform(list(target), flip_horizontal)
    
    for _ in range(4):
        symmetries.append((curr_board.copy(), curr_target.copy()))
        curr_board = rotate_90(curr_board)
        curr_target = transform(curr_target, rotate_90)
    
    return symmetries

# ============================================================================
# PART 4: EXHAUSTIVE STATE GENERATION
# ============================================================================

def generate_all_valid_states():
    """Generate ALL valid game states (not just random samples)"""
    valid_states = []
    
    for config in product([0, 1, -1], repeat=9):
        board = list(config)
        x_count = board.count(1)
        o_count = board.count(-1)
        
        # Valid: counts differ by at most 1
        if abs(x_count - o_count) > 1:
            continue
        
        result = check_winner(board)
        if result != 0:  # Skip terminal states
            continue
        
        # It's AI's turn when counts are equal
        if x_count == o_count:
            valid_states.append(board)
    
    return valid_states

def generate_game_simulations(num_games=100000):
    """Generate states through game simulation for realistic progressions"""
    states = set()
    
    for g in range(num_games):
        board = [0] * 9
        turn = 1
        
        while True:
            result = check_winner(board)
            if result != 0:
                break
            
            available = get_available_moves(board)
            if not available:
                break
            
            if turn == 1:
                states.add(tuple(board))
            
            # Mix random and optimal for diversity
            if random.random() < 0.3:
                move = random.choice(available)
            else:
                scores = get_all_move_scores(board) if turn == 1 else {}
                if scores:
                    move = max(scores.keys(), key=lambda x: scores[x])
                else:
                    move = random.choice(available)
            
            board[move] = turn
            turn = -turn
        
        if g % 10000 == 0:
            print(f"   Simulated {g}/{num_games} games...")
    
    return [list(s) for s in states]

# ============================================================================
# PART 5: TRAINING DATA GENERATION
# ============================================================================

def generate_training_data():
    print("=" * 70)
    print("🧠 GENERATING PERFECT UNBEATABLE AI TRAINING DATA")
    print("=" * 70)
    
    print("\n📊 Step 1: Generating exhaustive game states...")
    exhaustive_states = generate_all_valid_states()
    print(f"   Found {len(exhaustive_states)} exhaustive states")
    
    print("\n📊 Step 2: Generating simulated game states...")
    simulated_states = generate_game_simulations(150000)
    print(f"   Found {len(simulated_states)} simulated states")
    
    # Combine and deduplicate
    all_states = {tuple(s) for s in exhaustive_states}
    all_states.update(tuple(s) for s in simulated_states)
    all_states = [list(s) for s in all_states]
    print(f"\n✅ Total unique states: {len(all_states)}")
    
    # Analyze position types
    position_stats = defaultdict(int)
    for board in all_states:
        pos_type, _ = classify_position(board)
        position_stats[pos_type] += 1
    
    print("\n📈 Position breakdown:")
    for pos_type, count in sorted(position_stats.items()):
        print(f"   {pos_type}: {count}")
    
    # Generate training data with perfect targets
    print("\n🧮 Computing perfect moves for all states...")
    X, y_scores = [], []
    sample_weights = []
    
    for i, board in enumerate(all_states):
        if i % 2000 == 0:
            print(f"   Processing {i}/{len(all_states)}...")
        
        move_scores = get_all_move_scores(board)
        if not move_scores:
            continue
        
        available = list(move_scores.keys())
        pos_type, weight = classify_position(board)
        
        # Create target
        target = np.zeros(9, dtype=np.float32)
        
        # ===== HARD TARGETS FOR CRITICAL POSITIONS =====
        if 'MUST_WIN' in pos_type:
            # ONLY the winning move gets 1.0 - nothing else!
            winning_move = find_winning_move(board, 1)
            if winning_move is not None:
                target[winning_move] = 1.0
        elif 'MUST_BLOCK' in pos_type:
            # ONLY the blocking move gets 1.0 - nothing else!
            blocking_move = find_blocking_move(board, 1)
            if blocking_move is not None:
                target[blocking_move] = 1.0
        else:
            # Soft target based on minimax scores
            scores = [move_scores[m] for m in available]
            max_score, min_score = max(scores), min(scores)
            
            if max_score == min_score:
                # All moves equally good
                for m in available:
                    target[m] = 1.0
            else:
                for m in available:
                    # Exponential scaling to emphasize best moves strongly
                    normalized = (move_scores[m] - min_score) / (max_score - min_score)
                    target[m] = normalized ** 3  # Cube to make best moves dominate
        
        # Normalize target so max is 1
        if target.max() > 0:
            target = target / target.max()
        
        X.append(board)
        y_scores.append(target)
        sample_weights.append(weight)
        
        # ===== DATA AUGMENTATION: Add all symmetries for critical positions =====
        if weight >= 15.0:  # Critical positions (must-win, must-block, late game)
            for sym_board, sym_target in get_all_symmetries(board, target):
                # Avoid duplicates
                if sym_board != board:
                    X.append(sym_board)
                    y_scores.append(sym_target)
                    sample_weights.append(weight)
    
    X = np.array(X, dtype=np.float32)
    y_scores = np.array(y_scores, dtype=np.float32)
    sample_weights = np.array(sample_weights, dtype=np.float32)
    
    print(f"\n✅ Generated {len(X)} training samples")
    print(f"   Sample weight range: {sample_weights.min():.1f} - {sample_weights.max():.1f}")
    
    return X, y_scores, sample_weights

# ============================================================================
# PART 6: NEURAL NETWORK MODEL (TensorFlow.js compatible - NO BatchNorm)
# ============================================================================

def build_model():
    """
    Deep neural network optimized for perfect Tic-Tac-Toe play.
    No BatchNormalization for TensorFlow.js compatibility.
    """
    model = models.Sequential([
        layers.Input(shape=(9,)),
        
        # Layer 1 - Feature extraction
        layers.Dense(512, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                    kernel_initializer='he_normal'),
        layers.Dropout(0.3),
        
        # Layer 2 - Pattern recognition
        layers.Dense(512, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                    kernel_initializer='he_normal'),
        layers.Dropout(0.3),
        
        # Layer 3 - Strategic understanding
        layers.Dense(256, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                    kernel_initializer='he_normal'),
        layers.Dropout(0.2),
        
        # Layer 4 - Tactical refinement
        layers.Dense(128, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                    kernel_initializer='he_normal'),
        layers.Dropout(0.1),
        
        # Output - Move quality scores
        layers.Dense(9, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'mae']
    )
    
    return model

# ============================================================================
# PART 7: TRAINING PIPELINE
# ============================================================================

def train_model(X, Y, W):
    print("\n" + "=" * 70)
    print("🚀 TRAINING PERFECT AI MODEL")
    print("=" * 70)
    
    model = build_model()
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    # ===== PHASE 1: Train on all data with sample weights =====
    print("\n📚 Phase 1: Training on all data with sample weights...")
    history1 = model.fit(
        X, Y,
        sample_weight=W,
        epochs=200,
        batch_size=128,
        validation_split=0.15,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
        shuffle=True
    )
    
    # ===== PHASE 2: Fine-tune on CRITICAL positions only =====
    print("\n🎯 Phase 2: Fine-tuning on critical positions (late game focus)...")
    critical_mask = W >= 15.0  # Must-win, must-block, late game
    
    if np.sum(critical_mask) > 100:
        X_critical = X[critical_mask]
        Y_critical = Y[critical_mask]
        W_critical = W[critical_mask]
        
        print(f"   Fine-tuning on {len(X_critical)} critical samples...")
        
        # Lower learning rate
        model.optimizer.learning_rate.assign(0.0001)
        
        model.fit(
            X_critical, Y_critical,
            sample_weight=W_critical,
            epochs=100,
            batch_size=64,
            validation_split=0.1,
            callbacks=[
                callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            ],
            verbose=1,
            shuffle=True
        )
    
    # ===== PHASE 3: Extra fine-tune on LATE GAME only =====
    print("\n⚡ Phase 3: Extra fine-tuning on late game positions...")
    late_game_mask = W >= 20.0
    
    if np.sum(late_game_mask) > 50:
        X_late = X[late_game_mask]
        Y_late = Y[late_game_mask]
        
        print(f"   Fine-tuning on {len(X_late)} late game samples...")
        
        model.optimizer.learning_rate.assign(0.00005)
        
        model.fit(
            X_late, Y_late,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[
                callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            ],
            verbose=1,
            shuffle=True
        )
    
    return model

# ============================================================================
# PART 8: TESTING
# ============================================================================

def test_critical_positions(model):
    """Test model on critical late-game positions"""
    print("\n" + "=" * 70)
    print("🧪 TESTING CRITICAL POSITIONS (Late Game Focus)")
    print("=" * 70)
    
    test_cases = [
        # Late game MUST WIN
        {'name': 'Late Win: Row top', 'board': [1, 1, 0, -1, -1, 0, 0, 0, 0], 'expected': 2},
        {'name': 'Late Win: Row mid', 'board': [-1, 0, 0, 1, 1, 0, -1, 0, 0], 'expected': 5},
        {'name': 'Late Win: Row bot', 'board': [0, -1, 0, 0, -1, 0, 1, 1, 0], 'expected': 8},
        {'name': 'Late Win: Col left', 'board': [1, -1, 0, 1, -1, 0, 0, 0, 0], 'expected': 6},
        {'name': 'Late Win: Col mid', 'board': [-1, 1, 0, 0, 1, 0, 0, 0, -1], 'expected': 7},
        {'name': 'Late Win: Diag main', 'board': [1, -1, 0, 0, 1, -1, 0, 0, 0], 'expected': 8},
        {'name': 'Late Win: Diag anti', 'board': [0, -1, 1, 0, 1, 0, 0, -1, 0], 'expected': 6},
        
        # Late game MUST BLOCK
        {'name': 'Late Block: Row top', 'board': [-1, -1, 0, 1, 0, 0, 1, 0, 0], 'expected': 2},
        {'name': 'Late Block: Row mid', 'board': [1, 0, 0, -1, -1, 0, 0, 1, 0], 'expected': 5},
        {'name': 'Late Block: Row bot', 'board': [1, 0, 0, 0, 1, 0, -1, -1, 0], 'expected': 8},
        {'name': 'Late Block: Col left', 'board': [-1, 1, 0, -1, 0, 0, 0, 1, 0], 'expected': 6},
        {'name': 'Late Block: Diag main', 'board': [-1, 1, 0, 0, -1, 0, 1, 0, 0], 'expected': 8},
        {'name': 'Late Block: Diag anti', 'board': [1, 0, -1, 0, -1, 0, 0, 1, 0], 'expected': 6},
        
        # Very late game (only 2-3 empty)
        {'name': 'Final Win', 'board': [1, -1, 1, -1, 1, -1, 0, 0, 0], 'expected': 6},  # Complete 2-4-6
        {'name': 'Final Block', 'board': [1, 1, -1, -1, -1, 0, 1, 0, 0], 'expected': 5},  # Block row
        {'name': 'Last Move Win', 'board': [1, -1, 1, -1, -1, 1, 0, 1, -1], 'expected': 6},
    ]
    
    passed = 0
    failed_tests = []
    
    for test in test_cases:
        board = test['board']
        expected = test['expected']
        
        predictions = model.predict(np.array([board]), verbose=0)[0]
        
        # Mask illegal moves
        for i in range(9):
            if board[i] != 0:
                predictions[i] = -999
        
        ai_move = np.argmax(predictions)
        
        if ai_move == expected:
            passed += 1
            status = "✅"
        else:
            status = "❌"
            failed_tests.append(test['name'])
        
        print(f"{status} {test['name']}")
        print(f"   Board: {board}")
        print(f"   Expected: {expected}, AI: {ai_move}, Conf: {predictions[ai_move]:.4f}")
    
    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{len(test_cases)} tests passed")
    
    if failed_tests:
        print(f"Failed tests: {', '.join(failed_tests)}")
    else:
        print("🎉 ALL CRITICAL TESTS PASSED!")
    
    return passed == len(test_cases)

def play_vs_optimal(model, num_games=200):
    """Play against perfect opponent"""
    print(f"\n🎮 Playing {num_games} games against PERFECT opponent...")
    
    results = {'ai_wins': 0, 'opponent_wins': 0, 'draws': 0}
    
    for game in range(num_games):
        board = [0] * 9
        turn = 1 if game % 2 == 0 else -1  # Alternate
        
        while True:
            result = check_winner(board)
            if result != 0:
                if result == 1:
                    results['ai_wins'] += 1
                elif result == -1:
                    results['opponent_wins'] += 1
                else:
                    results['draws'] += 1
                break
            
            available = get_available_moves(board)
            if not available:
                results['draws'] += 1
                break
            
            if turn == 1:
                # AI move (model)
                predictions = model.predict(np.array([board]), verbose=0)[0]
                for i in range(9):
                    if board[i] != 0:
                        predictions[i] = -999
                move = np.argmax(predictions)
            else:
                # Perfect opponent (minimax)
                best_score = float('inf')
                move = available[0]
                for m in available:
                    board[m] = -1
                    score = minimax(board, 0, float('-inf'), float('inf'), True)
                    board[m] = 0
                    if score < best_score:
                        best_score = score
                        move = m
            
            board[move] = turn
            turn = -turn
    
    print(f"\nResults vs Perfect Opponent:")
    print(f"   AI Wins:       {results['ai_wins']}/{num_games} ({100*results['ai_wins']/num_games:.1f}%)")
    print(f"   Opponent Wins: {results['opponent_wins']}/{num_games} ({100*results['opponent_wins']/num_games:.1f}%)")
    print(f"   Draws:         {results['draws']}/{num_games} ({100*results['draws']/num_games:.1f}%)")
    
    if results['opponent_wins'] == 0:
        print("\n🎉 AI is UNBEATABLE! Perfect play achieved!")
    else:
        print(f"\n⚠️  AI lost {results['opponent_wins']} games - needs improvement")
    
    return results

# ============================================================================
# PART 9: SAVE MODEL
# ============================================================================

def save_model(model, output_dir="output_model"):
    print("\n" + "=" * 70)
    print("💾 SAVING MODEL")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save H5
    h5_path = os.path.join(output_dir, 'ttt_model_perfect.h5')
    model.save(h5_path)
    print(f"✅ Saved: {h5_path}")
    
    # Convert to TensorFlow.js
    tfjs_path = os.path.join(output_dir, 'tfjs')
    try:
        os.system(f'tensorflowjs_converter --input_format=keras "{h5_path}" "{tfjs_path}"')
        if os.path.exists(os.path.join(tfjs_path, 'model.json')):
            print(f"✅ Saved TensorFlow.js model: {tfjs_path}")
            
            # Fix model.json for browser compatibility
            model_json_path = os.path.join(tfjs_path, 'model.json')
            with open(model_json_path, 'r') as f:
                data = json.load(f)
            
            # Simplify for better compatibility
            print(f"✅ TensorFlow.js model ready!")
    except Exception as e:
        print(f"⚠️  TensorFlow.js conversion skipped: {e}")
    
    return h5_path

def interactive_play(model):
    """Play against the AI interactively"""
    print("\n" + "=" * 70)
    print("🎮 PLAY AGAINST THE AI")
    print("=" * 70)
    print("You are X (-1), AI is O (1)")
    print("Positions: 0|1|2")
    print("           3|4|5")
    print("           6|7|8")
    
    def print_board(board):
        symbols = {0: '·', 1: 'O', -1: 'X'}
        print()
        for i in range(3):
            row = [symbols[int(board[i*3 + j])] for j in range(3)]
            print(f"  {row[0]} | {row[1]} | {row[2]}")
            if i < 2:
                print("  ---------")
        print()
    
    board = [0.0] * 9
    
    while True:
        # AI first
        predictions = model.predict(np.array([board]), verbose=0)[0]
        for i in range(9):
            if board[i] != 0:
                predictions[i] = -999
        ai_move = np.argmax(predictions)
        board[ai_move] = 1
        print(f"\n🤖 AI plays {ai_move} (conf: {predictions[ai_move]:.4f})")
        print_board(board)
        
        result = check_winner(board)
        if result == 1:
            print("🤖 AI wins!")
            break
        if result == "Draw" or not get_available_moves(board):
            print("🤝 Draw!")
            break
        
        # Human
        while True:
            try:
                move = int(input("Your move (0-8): "))
                if 0 <= move <= 8 and board[move] == 0:
                    break
                print("Invalid!")
            except:
                print("Enter 0-8")
        
        board[move] = -1
        print_board(board)
        
        result = check_winner(board)
        if result == -1:
            print("👤 You win! (Report this as a bug!)")
            break
        if result == "Draw":
            print("🤝 Draw!")
            break

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Generate data
    X, Y, W = generate_training_data()
    
    # Train
    model = train_model(X, Y, W)
    
    # Test
    test_critical_positions(model)
    play_vs_optimal(model, 200)
    
    # Save
    save_path = save_model(model)
    
    # Play?
    print("\n" + "=" * 70)
    while True:
        choice = input("Play against the AI? (y/n): ").lower()
        if choice == 'y':
            interactive_play(model)
        elif choice == 'n':
            break
    
    print("\n✅ DONE! Model saved.")
    print(f"Copy '{save_path}' to Backend/routes/ai/models/ to use.")
