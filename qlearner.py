# Example usage for your GridWorldEnv
import gymnasium
import gymnasium_env
from gymnasium_env.wrappers import RelativePosition
import numpy as np
import random

class QLearner:
    def __init__(self, bins, n_actions):
        self.bins = bins  # Discretization bins
        self.n_actions = n_actions
        self.Q = np.zeros([np.prod([len(b) + 1 for b in bins]), n_actions])  # Q-table
        self.alpha = 0.8  # Learning rate
        self.discount_factor = 0.7  # Discount factor
        self.epsilon = 1  # Exploration rate

    def get_state_index(self, state):
        """Convert a discrete state tuple to a single index for the Q-table."""
        indices = [np.digitize(state[i], self.bins[i]) for i in range(len(self.bins))]
        return np.ravel_multi_index(indices, [len(b) + 1 for b in self.bins])

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        state_index = self.get_state_index(state)
        if random.uniform(0, 1) > self.epsilon:
            return np.argmax(self.Q[state_index, :])  # Exploit
        else:
            return random.randint(0, self.n_actions - 1)  # Explore

    def update_Q(self, state, new_state, action, reward):
        """Update the Q-table using the Bellman equation."""
        state_index = self.get_state_index(state)
        new_state_index = self.get_state_index(new_state)
        self.Q[state_index, action] += self.alpha * (
            reward + self.discount_factor * np.max(self.Q[new_state_index, :]) - self.Q[state_index, action]
        )

def flatten_state(obs):
    # Only use agent and target positions for state (can add more features if needed)
    return np.concatenate([obs["agent"], obs["target"]])

# Discretization bins for each coordinate (assuming grid size 5)
bins = [np.arange(5)] * 4  # agent_x, agent_y, target_x, target_y

env = gymnasium.make('gymnasium_env/GridWorld-v0', size=5)
qlearner = QLearner(bins, env.action_space.n)

n_episodes = 10
for episode in range(n_episodes):
    obs, _ = env.reset()
    state = flatten_state(obs)
    done = False
    while not done:
        action = qlearner.choose_action(state)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = flatten_state(next_obs)
        qlearner.update_Q(state, next_state, action, reward)
        state = next_state
        done = terminated or truncated
    # Decay epsilon
    qlearner.epsilon = max(0.05, qlearner.epsilon * 0.995)