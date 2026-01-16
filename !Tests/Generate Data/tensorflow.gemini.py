import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
import json
import os
import signal
import sys
from collections import deque
from datetime import datetime

# --- CHECKPOINT MANAGER ---
class CheckpointManager:
    def __init__(self, save_dir="checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.model_path = os.path.join(save_dir, "ttt_model.keras")
        self.memory_path = os.path.join(save_dir, "training_memory.json")
        self.metadata_path = os.path.join(save_dir, "metadata.json")
        
    def save_checkpoint(self, model, memory, agent_state):
        """Save model, memory, and agent state"""
        try:
            model.save(self.model_path)
            mem = [{"state": m[0].tolist(), "action": int(m[1]), "reward": float(m[2]), 
                    "next_state": m[3].tolist(), "done": bool(m[4])} for m in memory]
            with open(self.memory_path, 'w') as f:
                json.dump(mem, f)
            with open(self.metadata_path, 'w') as f:
                json.dump(agent_state, f)
            return True
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return False
    
    def load_checkpoint(self):
        """Load model, memory, and agent state if they exist"""
        if not all(os.path.exists(p) for p in [self.model_path, self.memory_path, self.metadata_path]):
            return None, None, None
        try:
            model = tf.keras.models.load_model(self.model_path)
            with open(self.memory_path, 'r') as f:
                mem_data = json.load(f)
            memory = deque([tuple(np.array(m["state"]) if isinstance(m["state"], list) else m["state"],
                                   m["action"], m["reward"],
                                   np.array(m["next_state"]) if isinstance(m["next_state"], list) else m["next_state"],
                                   m["done"]) for m in mem_data], maxlen=2000)
            with open(self.metadata_path, 'r') as f:
                agent_state = json.load(f)
            return model, memory, agent_state
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None, None, None
    
    def has_checkpoint(self):
        return all(os.path.exists(p) for p in [self.model_path, self.memory_path, self.metadata_path])

# --- 1. THE ENVIRONMENT ---
class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9)
    
    def reset(self):
        self.board = np.zeros(9)
        return self.board.copy()

    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x == 0]

    def step(self, action, player):
        if self.board[action] != 0:
            return self.board.copy(), -10, True 
        self.board[action] = player
        reward, done = self.check_winner(player)
        return self.board.copy(), reward, done

    def check_winner(self, player):
        win_states = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for line in win_states:
            if self.board[line[0]] == self.board[line[1]] == self.board[line[2]] == player:
                return 1, True
        if len(self.available_moves()) == 0:
            return 0.5, True
        return 0, False

# --- 2. THE AI AGENT ---
class DQNAgent:
    def __init__(self, checkpoint_model=None, checkpoint_memory=None, checkpoint_state=None):
        self.memory = checkpoint_memory if checkpoint_memory is not None else deque(maxlen=6000)
        self.gamma = 0.95    
        self.epsilon = 1.0 if checkpoint_state is None else checkpoint_state.get("epsilon", 1.0)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9985
        self.learning_rate = 0.0005
        self.loss_history = []
        
        if checkpoint_model is not None:
            self.model = checkpoint_model
        else:
            self.model = self._build_model()

    def _build_model(self):
        """Deeper network for better learning"""
        model = tf.keras.Sequential([
            layers.Input(shape=(9,)),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(9, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state, available_moves, training=True):
        current_epsilon = self.epsilon if training else 0.0
        if np.random.rand() <= current_epsilon:
            return random.choice(available_moves)
        act_values = self.model.predict(state.reshape(1, 9), verbose=0)
        valid_acts = [(act_values[0][i], i) for i in available_moves]
        return max(valid_acts, key=lambda x: x[0])[1]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([m[0] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            targets[i][action] = reward if done else reward + self.gamma * np.max(next_q_values[i])
        
        history = self.model.fit(states, targets, epochs=1, verbose=0, batch_size=32)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss

    def get_state(self):
        """Get agent state for checkpointing"""
        return {
            "epsilon": float(self.epsilon),
            "loss_history": self.loss_history[-100:] if len(self.loss_history) > 100 else self.loss_history
        }

# --- 3. THE EVALUATOR ---
def evaluate_ai(agent, env, episodes=100):
    """Plays games without randomness to see how good the AI actually is."""
    wins = 0
    draws = 0
    losses = 0
    
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, env.available_moves(), training=False)
            state, reward, done = env.step(action, 1)
            if done:
                if reward == 1:
                    wins += 1
                elif reward == 0.5:
                    draws += 1
                else:
                    losses += 1
            if not done:
                _, _, done = env.step(random.choice(env.available_moves()), -1)
                if done:
                    losses += 1
    
    return {
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": (wins / episodes) * 100,
        "draw_rate": (draws / episodes) * 100
    }

def graceful_shutdown(signum, frame, checkpoint_manager, agent, env):
    """Handle Ctrl+C gracefully by saving before exit"""
    print("\n\n🛑 Graceful shutdown triggered. Saving checkpoint...")
    agent_state = agent.get_state()
    if checkpoint_manager.save_checkpoint(agent.model, agent.memory, agent_state):
        print("✅ Emergency checkpoint saved successfully!")
    sys.exit(0)

# --- 4. CONTINUOUS TRAINING LOOP ---
def main():
    print("="*80)
    print("TIC TAC TOE AI TRAINING - LONG TERM LEARNING")
    print("="*80)
    
    checkpoint_manager = CheckpointManager()
    env = TicTacToe()
    
    start_episode = 1
    if checkpoint_manager.has_checkpoint():
        print("\n📂 Found existing checkpoint. Loading...")
        model, memory, agent_state = checkpoint_manager.load_checkpoint()
        if model is not None:
            agent = DQNAgent(checkpoint_model=model, checkpoint_memory=memory, checkpoint_state=agent_state)
            # Try to resume from metadata if available
            try:
                with open(os.path.join(checkpoint_manager.save_dir, "episode_counter.txt"), 'r') as f:
                    start_episode = int(f.read()) + 1
                print(f"✅ Resumed from episode {start_episode}")
            except:
                print("Starting fresh training with loaded model")
        else:
            print("⚠️  Failed to load checkpoint. Starting fresh...")
            agent = DQNAgent()
    else:
        print("\n🆕 No checkpoint found. Starting fresh training...")
        agent = DQNAgent()
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, lambda s, f: graceful_shutdown(s, f, checkpoint_manager, agent, env))
    
    total_episodes = 1000000  # Essentially infinite
    save_every = 20
    eval_every = 50
    batch_size = 128  # Larger batch size for better learning
    
    print(f"\n{'Episode':<10} | {'Win %':<8} | {'Draw %':<8} | {'Loss %':<8} | {'Memory':<8} | {'Epsilon':<10} | {'Avg Loss':<10}")
    print("-" * 95)
    
    try:
        for e in range(start_episode, start_episode + total_episodes):
            state = env.reset()
            done = False
            episode_loss = 0
            training_steps = 0
            
            while not done:
                action = agent.act(state, env.available_moves())
                next_state, reward, done = env.step(action, 1)
                
                if not done:
                    opp_move = random.choice(env.available_moves())
                    next_state, opp_reward, done = env.step(opp_move, -1)
                    if done and opp_reward == 1:
                        reward = -1 

                agent.remember(state, action, reward, next_state, done)
                state = next_state
            
            # Train multiple times per episode for better convergence
            if len(agent.memory) > batch_size:
                for _ in range(2):  # Multiple training passes
                    loss = agent.replay(batch_size)
                    episode_loss += loss
                    training_steps += 1
            
            # Evaluate periodically
            if e % eval_every == 0:
                results = evaluate_ai(agent, env, episodes=100)
                avg_loss = episode_loss / training_steps if training_steps > 0 else 0
                print(f"{e:<10} | {results['win_rate']:<8.1f} | {results['draw_rate']:<8.1f} | "
                      f"{(100 - results['win_rate'] - results['draw_rate']):<8.1f} | {len(agent.memory):<8} | "
                      f"{agent.epsilon:<10.4f} | {avg_loss:<10.6f}")
            
            # Save checkpoint periodically
            if e % save_every == 0:
                agent_state = agent.get_state()
                if checkpoint_manager.save_checkpoint(agent.model, agent.memory, agent_state):
                    with open(os.path.join(checkpoint_manager.save_dir, "episode_counter.txt"), 'w') as f:
                        f.write(str(e))
    
    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted. Saving...")
        agent_state = agent.get_state()
        if checkpoint_manager.save_checkpoint(agent.model, agent.memory, agent_state):
            print("✅ Checkpoint saved!")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        print("Attempting emergency save...")
        agent_state = agent.get_state()
        if checkpoint_manager.save_checkpoint(agent.model, agent.memory, agent_state):
            print("✅ Emergency checkpoint saved!")
        raise

if __name__ == "__main__":
    main()