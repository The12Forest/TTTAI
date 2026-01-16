import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
import json
import os
from collections import deque

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
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    
        self.epsilon = 1.0   
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            layers.Input(shape=(9,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(9, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def act(self, state, available_moves, training=True):
        # If not training, we use 0 epsilon (pure intelligence, no guessing)
        current_epsilon = self.epsilon if training else 0.0
        if np.random.rand() <= current_epsilon:
            return random.choice(available_moves)
        act_values = self.model.predict(state.reshape(1, 9), verbose=0)
        valid_acts = [(act_values[0][i], i) for i in available_moves]
        return max(valid_acts, key=lambda x: x[0])[1]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size: return
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([m[0] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            targets[i][action] = reward if done else reward + self.gamma * np.max(next_q_values[i])
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_all(self):
        self.model.save("ttt_model.keras")
        mem = [{"state": m[0].tolist(), "action": int(m[1]), "reward": float(m[2]), 
                "next_state": m[3].tolist(), "done": bool(m[4])} for m in self.memory]
        with open("training_data.json", 'w') as f:
            json.dump(mem, f)

# --- 3. THE EVALUATOR ---
def evaluate_ai(agent, env, episodes=20):
    """Plays games without randomness to see how good the AI actually is."""
    wins = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state, env.available_moves(), training=False)
            state, reward, done = env.step(action, 1)
            if done and reward == 1:
                wins += 1
            if not done:
                _, _, done = env.step(random.choice(env.available_moves()), -1)
    return (wins / episodes) * 100

# --- 4. CONTINUOUS TRAINING LOOP ---
env = TicTacToe()
agent = DQNAgent()

total_episodes = 1000
save_every = 50
eval_every = 50

print(f"{'Episode':<10} | {'Win Rate %':<12} | {'Epsilon':<10}")
print("-" * 35)

for e in range(1, total_episodes + 1):
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state, env.available_moves())
        next_state, reward, done = env.step(action, 1)
        
        if not done:
            opp_move = random.choice(env.available_moves())
            next_state, opp_reward, done = env.step(opp_move, -1)
            if done and opp_reward == 1: reward = -1 

        agent.remember(state, action, reward, next_state, done)
        state = next_state
    
    if len(agent.memory) > 32:
        agent.replay(32)

    # Every 50 episodes: Evaluate and Save
    if e % eval_every == 0:
        win_rate = evaluate_ai(agent, env)
        print(f"{e:<10} | {win_rate:<12.1f} | {agent.epsilon:<10.3f}")
        agent.save_all()

print("\nTraining Finished. Files 'ttt_model.keras' and 'training_data.json' updated.")