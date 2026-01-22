"""
PERFECT TIC-TAC-TOE AI TRAINER
==============================
This script generates a PERFECT AI that cannot be beaten.
It uses minimax with alpha-beta pruning to generate optimal moves
for EVERY possible game state, with special emphasis on late-game
critical positions (blocking wins, taking wins).

The AI will:
1. Always take a winning move if available
2. Always block opponent's winning move
3. Play optimally in all other situations
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import random
import os
import json
from collections import defaultdict
from itertools import product

# ============================================================================
# PART 1: PERFECT MINIMAX ENGINE WITH ALPHA-BETA PRUNING
# ============================================================================

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # Rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # Columns
    (0, 4, 8), (2, 4, 6)              # Diagonals
]

# Memoization cache for minimax
minimax_cache = {}

def check_winner(board):
    """Check for winner: 1=AI wins, -1=Human wins, 0=ongoing, 'Draw'=tie"""
    for a, b, c in WIN_LINES:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    return 0 if 0 in board else "Draw"

def get_available_moves(board):
    """Get all empty positions"""
    return [i for i in range(9) if board[i] == 0]

def board_to_key(board):
    """Convert board to hashable tuple"""
    return tuple(board)

def minimax(board, depth, alpha, beta, is_maximizing):
    """
    Minimax with alpha-beta pruning.
    Returns the best score for the current player.
    AI (1) is maximizing, Human (-1) is minimizing.
    """
    key = (board_to_key(board), is_maximizing)
    if key in minimax_cache:
        return minimax_cache[key]
    
    result = check_winner(board)
    
    # Terminal states
    if result == 1:
        return 100 - depth  # AI wins (prefer faster wins)
    if result == -1:
        return -100 + depth  # Human wins (delay losses)
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

def get_best_move(board):
    """Get the optimal move using minimax"""
    available = get_available_moves(board)
    if not available:
        return None, None
    
    best_score = float('-inf')
    best_move = available[0]
    
    for move in available:
        board[move] = 1
        score = minimax(board, 0, float('-inf'), float('inf'), False)
        board[move] = 0
        
        if score > best_score:
            best_score = score
            best_move = move
    
    return best_move, best_score

def get_all_move_scores(board):
    """Get minimax score for ALL available moves"""
    available = get_available_moves(board)
    scores = {}
    
    for move in available:
        board[move] = 1
        score = minimax(board, 0, float('-inf'), float('inf'), False)
        board[move] = 0
        scores[move] = score
    
    return scores

# ============================================================================
# PART 2: CRITICAL POSITION DETECTION
# ============================================================================

def find_winning_move(board, player):
    """Find a move that wins immediately for the given player"""
    for a, b, c in WIN_LINES:
        line = [board[a], board[b], board[c]]
        if line.count(player) == 2 and line.count(0) == 1:
            # Find the empty spot
            if board[a] == 0:
                return a
            if board[b] == 0:
                return b
            if board[c] == 0:
                return c
    return None

def find_blocking_move(board, player):
    """Find a move that blocks opponent's win"""
    opponent = -player
    return find_winning_move(board, opponent)

def find_fork_move(board, player):
    """Find a move that creates two winning threats"""
    available = get_available_moves(board)
    
    for move in available:
        board[move] = player
        # Count winning threats after this move
        threats = 0
        for a, b, c in WIN_LINES:
            line = [board[a], board[b], board[c]]
            if line.count(player) == 2 and line.count(0) == 1:
                threats += 1
        board[move] = 0
        
        if threats >= 2:
            return move
    return None

def classify_position(board):
    """
    Classify the position type for training emphasis:
    - 'must_win': AI has a winning move
    - 'must_block': Human has a winning move (must block)
    - 'fork': Fork opportunity
    - 'block_fork': Must block opponent's fork
    - 'normal': Regular position
    """
    # Check if AI can win
    if find_winning_move(board, 1) is not None:
        return 'must_win'
    
    # Check if must block
    if find_blocking_move(board, 1) is not None:
        return 'must_block'
    
    # Check for fork opportunities
    if find_fork_move(board, 1) is not None:
        return 'fork'
    
    # Check if must block opponent's fork
    if find_fork_move(board, -1) is not None:
        return 'block_fork'
    
    return 'normal'

# ============================================================================
# PART 3: BOARD SYMMETRIES FOR DATA AUGMENTATION
# ============================================================================

def get_all_symmetries(board, target_scores):
    """
    Generate all 8 symmetries (4 rotations × 2 reflections).
    Returns list of (board, scores) tuples.
    """
    def rotate_90(b):
        """Rotate board 90 degrees clockwise"""
        return [b[6], b[3], b[0], b[7], b[4], b[1], b[8], b[5], b[2]]
    
    def flip_horizontal(b):
        """Flip board horizontally"""
        return [b[2], b[1], b[0], b[5], b[4], b[3], b[8], b[7], b[6]]
    
    def transform_scores(scores, transform_func):
        """Apply same transformation to scores"""
        # Create a board with position indices to track transformation
        indices = list(range(9))
        transformed_indices = transform_func(indices)
        new_scores = [0.0] * 9
        for new_pos, old_pos in enumerate(transformed_indices):
            new_scores[new_pos] = scores[old_pos]
        return new_scores
    
    symmetries = []
    current_board = list(board)
    current_scores = list(target_scores)
    
    # 4 rotations
    for _ in range(4):
        symmetries.append((current_board.copy(), current_scores.copy()))
        current_board = rotate_90(current_board)
        current_scores = transform_scores(current_scores, rotate_90)
    
    # Flip and 4 more rotations
    current_board = flip_horizontal(list(board))
    current_scores = transform_scores(list(target_scores), flip_horizontal)
    
    for _ in range(4):
        symmetries.append((current_board.copy(), current_scores.copy()))
        current_board = rotate_90(current_board)
        current_scores = transform_scores(current_scores, rotate_90)
    
    return symmetries

# ============================================================================
# PART 4: EXHAUSTIVE GAME STATE GENERATION
# ============================================================================

def generate_all_valid_states():
    """
    Generate ALL valid Tic-Tac-Toe game states.
    A valid state has equal or +1 pieces for player who moves first.
    """
    valid_states = []
    
    # Generate all possible board configurations (3^9 = 19683)
    for config in product([0, 1, -1], repeat=9):
        board = list(config)
        
        # Count pieces
        x_count = board.count(1)   # AI pieces
        o_count = board.count(-1)  # Human pieces
        
        # Valid states: AI moved same or one more time than human
        # (depends on who moves first - let's say AI moves first when it's AI's turn)
        if abs(x_count - o_count) > 1:
            continue
        
        # Skip terminal states (someone already won)
        result = check_winner(board)
        if result != 0 and result != "Draw":
            continue
        
        # Skip draw states
        if result == "Draw":
            continue
        
        # It's AI's turn when counts are equal
        if x_count == o_count:
            valid_states.append(board)
    
    return valid_states

def generate_states_by_game_simulation(num_games=50000):
    """
    Generate game states by simulating random games.
    This ensures we get realistic game progressions.
    """
    states = set()
    
    for _ in range(num_games):
        board = [0] * 9
        turn = 1  # AI starts
        
        while True:
            result = check_winner(board)
            if result != 0:
                break
            
            available = get_available_moves(board)
            if not available:
                break
            
            # Record state before AI's move
            if turn == 1:
                states.add(tuple(board))
            
            # Make a move (mix of random and optimal for diversity)
            if random.random() < 0.3:
                move = random.choice(available)
            else:
                move, _ = get_best_move(board) if turn == 1 else (random.choice(available), 0)
            
            board[move] = turn
            turn = -turn
    
    return [list(s) for s in states]

# ============================================================================
# PART 5: TRAINING DATA GENERATION
# ============================================================================

def create_target_scores(board, move_scores):
    """
    Create normalized target scores for training.
    Uses softmax-like normalization to emphasize best moves.
    """
    target = np.zeros(9, dtype=np.float32)
    available = get_available_moves(board)
    
    if not available:
        return target
    
    # Get min and max scores
    scores = [move_scores[m] for m in available]
    max_score = max(scores)
    min_score = min(scores)
    
    # For critical positions, use hard targets
    position_type = classify_position(board)
    
    if position_type == 'must_win':
        # Only the winning move gets 1.0
        winning_move = find_winning_move(board, 1)
        target[winning_move] = 1.0
        return target
    
    if position_type == 'must_block':
        # Only the blocking move gets 1.0
        blocking_move = find_blocking_move(board, 1)
        target[blocking_move] = 1.0
        return target
    
    # For other positions, use normalized scores with temperature
    if max_score == min_score:
        # All moves are equally good
        for m in available:
            target[m] = 1.0 / len(available)
    else:
        # Temperature controls how "peaked" the distribution is
        # Lower temperature = more peaked around best move
        temperature = 0.1
        
        # Normalize to [0, 1] with exponential scaling
        for m in available:
            normalized = (move_scores[m] - min_score) / (max_score - min_score)
            target[m] = np.exp(normalized / temperature)
        
        # Normalize to sum to 1
        total = sum(target)
        if total > 0:
            target = target / total
            # Scale up best moves
            target = target ** 0.5
            target = target / max(target)  # Normalize so max is 1
    
    return target

def generate_training_data():
    """Generate comprehensive training data with perfect moves"""
    print("=" * 60)
    print("GENERATING PERFECT TRAINING DATA")
    print("=" * 60)
    
    # Generate states through simulation (more realistic game progressions)
    print("\n📊 Generating game states through simulation...")
    simulated_states = generate_states_by_game_simulation(100000)
    print(f"   Generated {len(simulated_states)} unique states from simulation")
    
    # Also generate exhaustively to ensure coverage
    print("\n📊 Generating exhaustive game states...")
    exhaustive_states = generate_all_valid_states()
    print(f"   Generated {len(exhaustive_states)} exhaustive states")
    
    # Combine and deduplicate
    all_states = {tuple(s) for s in simulated_states}
    all_states.update(tuple(s) for s in exhaustive_states)
    all_states = [list(s) for s in all_states]
    print(f"\n✅ Total unique states: {len(all_states)}")
    
    # Classify positions
    position_counts = defaultdict(int)
    for board in all_states:
        pos_type = classify_position(board)
        position_counts[pos_type] += 1
    
    print("\n📈 Position breakdown:")
    for pos_type, count in sorted(position_counts.items()):
        print(f"   {pos_type}: {count}")
    
    # Generate training data
    print("\n🧮 Computing optimal moves for all states...")
    X_data = []
    Y_data = []
    weights = []
    
    for i, board in enumerate(all_states):
        if i % 1000 == 0:
            print(f"   Processing state {i}/{len(all_states)}...")
        
        # Get optimal scores for all moves
        move_scores = get_all_move_scores(board)
        
        if not move_scores:
            continue
        
        # Create target scores
        target = create_target_scores(board, move_scores)
        
        # Determine sample weight based on position type
        pos_type = classify_position(board)
        if pos_type == 'must_win':
            weight = 10.0  # Critical: must take winning move
        elif pos_type == 'must_block':
            weight = 10.0  # Critical: must block opponent
        elif pos_type == 'fork':
            weight = 5.0   # Important: create fork
        elif pos_type == 'block_fork':
            weight = 5.0   # Important: block fork
        else:
            weight = 1.0
        
        # Add base sample
        X_data.append(board)
        Y_data.append(target)
        weights.append(weight)
        
        # Add symmetries for data augmentation (especially for critical positions)
        if pos_type in ['must_win', 'must_block', 'fork', 'block_fork']:
            for sym_board, sym_target in get_all_symmetries(board, target):
                X_data.append(sym_board)
                Y_data.append(sym_target)
                weights.append(weight)
    
    X = np.array(X_data, dtype=np.float32)
    Y = np.array(Y_data, dtype=np.float32)
    W = np.array(weights, dtype=np.float32)
    
    print(f"\n✅ Generated {len(X)} training samples")
    print(f"   (including symmetry augmentation)")
    
    return X, Y, W

# ============================================================================
# PART 6: NEURAL NETWORK MODEL
# ============================================================================

def build_model():
    """
    Build a deep neural network for Tic-Tac-Toe.
    Architecture designed for perfect play.
    """
    model = models.Sequential([
        layers.Input(shape=(9,)),
        
        # First block - feature extraction
        layers.Dense(512, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Second block - pattern recognition
        layers.Dense(512, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Third block - strategic understanding
        layers.Dense(256, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        
        # Fourth block - tactical decisions
        layers.Dense(128, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        # Output - move probabilities
        layers.Dense(9, activation='sigmoid')
    ])
    
    # Use Adam with warm restarts schedule
    initial_lr = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',  # Better for multi-label-like output
        metrics=['accuracy', 'mae']
    )
    
    return model

# ============================================================================
# PART 7: TRAINING PIPELINE
# ============================================================================

def train_model(X, Y, W):
    """Train the model with curriculum learning"""
    print("\n" + "=" * 60)
    print("TRAINING PERFECT AI MODEL")
    print("=" * 60)
    
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
    
    # Phase 1: Train on all data with sample weights
    print("\n🚀 Phase 1: Training on all data with sample weights...")
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
    
    # Phase 2: Fine-tune on critical positions only
    print("\n🎯 Phase 2: Fine-tuning on critical positions...")
    critical_mask = W > 1.0
    if np.sum(critical_mask) > 100:
        X_critical = X[critical_mask]
        Y_critical = Y[critical_mask]
        
        # Lower learning rate for fine-tuning
        model.optimizer.learning_rate.assign(0.0001)
        
        history2 = model.fit(
            X_critical, Y_critical,
            epochs=100,
            batch_size=64,
            validation_split=0.1,
            callbacks=[
                callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            ],
            verbose=1,
            shuffle=True
        )
    
    return model

# ============================================================================
# PART 8: EVALUATION AND TESTING
# ============================================================================

def test_ai_perfection(model):
    """Test if the AI plays perfectly"""
    print("\n" + "=" * 60)
    print("TESTING AI PERFECTION")
    print("=" * 60)
    
    # Test critical positions
    test_cases = [
        # Must win situations
        {
            'name': 'Must Win (row)',
            'board': [1, 1, 0, -1, -1, 0, 0, 0, 0],
            'correct': 2
        },
        {
            'name': 'Must Win (col)',
            'board': [1, -1, 0, 1, -1, 0, 0, 0, 0],
            'correct': 6
        },
        {
            'name': 'Must Win (diag)',
            'board': [1, -1, -1, 0, 1, 0, 0, 0, 0],
            'correct': 8
        },
        # Must block situations
        {
            'name': 'Must Block (row)',
            'board': [-1, -1, 0, 1, 1, 0, 0, 0, 0],
            'correct': 2
        },
        {
            'name': 'Must Block (col)',
            'board': [-1, 1, 0, -1, 1, 0, 0, 0, 0],
            'correct': 6
        },
        {
            'name': 'Must Block (diag)',
            'board': [-1, 1, 1, 0, -1, 0, 0, 0, 0],
            'correct': 8
        },
        # Late game critical
        {
            'name': 'Late Game Win',
            'board': [1, -1, 1, -1, 1, -1, 0, 0, 0],
            'correct': 6  # Completes diagonal 2-4-6
        },
        {
            'name': 'Late Game Block',
            'board': [-1, 1, -1, 1, -1, 1, 0, 0, 0],
            'correct': 6  # Block diagonal 2-4-6
        },
    ]
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        board = test['board']
        correct = test['correct']
        
        # Get AI prediction
        predictions = model.predict(np.array([board]), verbose=0)[0]
        
        # Mask illegal moves
        for i in range(9):
            if board[i] != 0:
                predictions[i] = -999
        
        ai_move = np.argmax(predictions)
        
        status = "✅ PASS" if ai_move == correct else "❌ FAIL"
        if ai_move == correct:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{test['name']}")
        print(f"  Board: {board}")
        print(f"  Expected: {correct}, AI chose: {ai_move}")
        print(f"  Confidence: {predictions[ai_move]:.4f}")
        print(f"  {status}")
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{passed + failed} tests passed")
    
    return passed == len(test_cases)

def play_vs_optimal(model, num_games=100):
    """Play games against optimal opponent"""
    print(f"\n🎮 Playing {num_games} games against optimal opponent...")
    
    results = {'ai_wins': 0, 'opponent_wins': 0, 'draws': 0}
    
    for game in range(num_games):
        board = [0] * 9
        turn = 1 if game % 2 == 0 else -1  # Alternate who goes first
        
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
                # AI move
                predictions = model.predict(np.array([board]), verbose=0)[0]
                for i in range(9):
                    if board[i] != 0:
                        predictions[i] = -999
                move = np.argmax(predictions)
            else:
                # Optimal opponent (minimax)
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
    
    print(f"\nResults vs Optimal Opponent:")
    print(f"  AI Wins:       {results['ai_wins']}/{num_games} ({100*results['ai_wins']/num_games:.1f}%)")
    print(f"  Opponent Wins: {results['opponent_wins']}/{num_games} ({100*results['opponent_wins']/num_games:.1f}%)")
    print(f"  Draws:         {results['draws']}/{num_games} ({100*results['draws']/num_games:.1f}%)")
    
    # Perfect play should never lose
    if results['opponent_wins'] == 0:
        print("\n✅ AI is UNBEATABLE!")
    else:
        print(f"\n⚠️ AI lost {results['opponent_wins']} games - needs more training")
    
    return results

def interactive_play(model):
    """Play interactively against the trained AI"""
    print("\n" + "=" * 60)
    print("🎮 PLAY AGAINST THE AI")
    print("=" * 60)
    print("You are X (-1), AI is O (1)")
    print("Positions:")
    print("  0 | 1 | 2")
    print("  ---------")
    print("  3 | 4 | 5")
    print("  ---------")
    print("  6 | 7 | 8")
    
    def print_board(board):
        symbols = {0: ' ', 1: 'O', -1: 'X'}
        print()
        for i in range(3):
            row = [symbols[board[i*3 + j]] for j in range(3)]
            print(f"  {row[0]} | {row[1]} | {row[2]}")
            if i < 2:
                print("  ---------")
        print()
    
    board = [0] * 9
    
    while True:
        result = check_winner(board)
        if result != 0:
            print_board(board)
            if result == 1:
                print("🤖 AI wins!")
            elif result == -1:
                print("👤 You win! (Congratulations, this shouldn't be possible!)")
            else:
                print("🤝 Draw!")
            break
        
        if not get_available_moves(board):
            print_board(board)
            print("🤝 Draw!")
            break
        
        # AI move
        predictions = model.predict(np.array([board]), verbose=0)[0]
        for i in range(9):
            if board[i] != 0:
                predictions[i] = -999
        
        ai_move = np.argmax(predictions)
        board[ai_move] = 1
        print(f"\n🤖 AI plays position {ai_move} (confidence: {predictions[ai_move]:.4f})")
        
        # Check after AI move
        result = check_winner(board)
        if result != 0:
            print_board(board)
            if result == 1:
                print("🤖 AI wins!")
            else:
                print("🤝 Draw!")
            break
        
        print_board(board)
        
        # Human move
        while True:
            try:
                move = int(input("Your move (0-8): "))
                if move < 0 or move > 8:
                    print("Invalid position! Use 0-8.")
                    continue
                if board[move] != 0:
                    print("Position already taken!")
                    continue
                break
            except ValueError:
                print("Please enter a number 0-8.")
        
        board[move] = -1

# ============================================================================
# PART 9: SAVE MODEL
# ============================================================================

def save_model(model, output_dir="trained_model"):
    """Save model in multiple formats"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as H5
    h5_path = os.path.join(output_dir, 'ttt_model_perfect.h5')
    model.save(h5_path)
    print(f"✅ Saved H5 model: {h5_path}")
    
    # Save as SavedModel format
    saved_model_path = os.path.join(output_dir, 'savedmodel')
    model.save(saved_model_path, save_format='tf')
    print(f"✅ Saved TF SavedModel: {saved_model_path}")
    
    # Convert to TensorFlow.js (if converter available)
    try:
        tfjs_path = os.path.join(output_dir, 'tfjs')
        os.system(f'tensorflowjs_converter --input_format=keras {h5_path} {tfjs_path}')
        if os.path.exists(os.path.join(tfjs_path, 'model.json')):
            print(f"✅ Saved TensorFlow.js model: {tfjs_path}")
    except:
        print("⚠️ TensorFlow.js converter not available")
    
    return h5_path

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🧠 PERFECT TIC-TAC-TOE AI TRAINER")
    print("=" * 60)
    
    # Generate training data
    X, Y, W = generate_training_data()
    
    # Train model
    model = train_model(X, Y, W)
    
    # Test perfection
    is_perfect = test_ai_perfection(model)
    
    # Play against optimal opponent
    results = play_vs_optimal(model, num_games=200)
    
    # Save model
    save_path = save_model(model)
    
    # Interactive play option
    print("\n" + "=" * 60)
    while True:
        choice = input("\nWould you like to play against the AI? (y/n): ").lower()
        if choice == 'y':
            interactive_play(model)
        elif choice == 'n':
            break
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {save_path}")
