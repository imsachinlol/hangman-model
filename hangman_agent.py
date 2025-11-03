import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, defaultdict
import pickle

class DQN(nn.Module):
    """Deep Q-Network for Hangman."""
    
    def __init__(self, state_size=55, action_size=26):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class ReplayMemory:
    """Experience replay buffer."""
    
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Save a transition."""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class HangmanDQNAgent:
    """DQN Agent for Hangman."""
    
    def __init__(self, state_size=55, action_size=26, learning_rate=0.0005,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995,
                 memory_size=10000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Main network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        
        # Target network
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = ReplayMemory(memory_size)
        
        self.steps = 0
        self.target_update_freq = 100
        
    def update_target_model(self):
        """Copy weights from main model to target model."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def act(self, state, valid_actions, use_hmm_probs=True):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current game state (dict with 'hmm_probs' and other info)
            valid_actions: List of valid action letters
            use_hmm_probs: Whether to combine Q-values with HMM probabilities
            
        Returns:
            action: Letter to guess
        """
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Get state vector for neural network
        # Need to create state vector from state dict
        state_vector = self._state_to_vector(state)
        
        # Get Q-values from model
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor).cpu().numpy()[0]
        
        # Combine with HMM probabilities if available
        if use_hmm_probs and 'hmm_probs' in state:
            hmm_probs = state['hmm_probs']
            # Normalize Q-values
            q_norm = (q_values - q_values.min()) / (q_values.max() - q_values.min() + 1e-8)
            # Weighted combination
            combined = 0.7 * q_norm + 0.3 * hmm_probs * 100
        else:
            combined = q_values
        
        # Mask invalid actions
        valid_indices = [ord(a) - ord('a') for a in valid_actions]
        masked_values = np.full(26, -np.inf)
        masked_values[valid_indices] = combined[valid_indices]
        
        # Choose best action
        action_idx = np.argmax(masked_values)
        return chr(action_idx + ord('a'))
    
    def _state_to_vector(self, state):
        """Convert state dict to vector for neural network."""
        # Guessed letters (26 binary features)
        guessed_vec = np.zeros(26)
        for letter in state['guessed_letters']:
            idx = ord(letter) - ord('a')
            guessed_vec[idx] = 1
        
        # Lives remaining (normalized)
        lives_vec = np.array([state['lives_remaining'] / 6.0])
        
        # HMM probabilities (26 features)
        hmm_vec = state.get('hmm_probs', np.zeros(26))
        
        # Word length (normalized)
        length_vec = np.array([state['word_length'] / 20.0])
        
        # Number of blanks (normalized)
        blanks_vec = np.array([state['num_blanks'] / state['word_length']])
        
        # Concatenate
        state_vector = np.concatenate([
            guessed_vec,
            lives_vec,
            hmm_vec,
            length_vec,
            blanks_vec
        ])
        
        return state_vector
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        state_vec = self._state_to_vector(state)
        next_state_vec = self._state_to_vector(next_state)
        action_idx = ord(action) - ord('a')
        self.memory.push(state_vec, action_idx, reward, next_state_vec, done)
    
    def replay(self):
        """Train on batch from replay memory."""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        current_q = self.model(states).gather(1, actions).squeeze()
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = self.criterion(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_model()
        
        return loss.item()
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filename):
        """Save model to file."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }
        torch.save(checkpoint, filename)
        print(f"DQN model saved to {filename}")
    
    def load(self, filename):
        """Load model from file."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.steps = checkpoint.get('steps', 0)
        print(f"DQN model loaded from {filename}")


class SimpleQLearningAgent:
    """Table-based Q-Learning Agent for Hangman."""
    
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.05, epsilon_decay=0.9995):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table: dict of (state_key, action) -> Q-value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        self.action_space = 'abcdefghijklmnopqrstuvwxyz'
    
    def _get_state_key(self, state):
        """Convert state to hashable key for Q-table."""
        masked_word = state['masked_word']
        guessed = ''.join(sorted(state['guessed_letters']))
        lives = state['lives_remaining']
        return f"{masked_word}:{guessed}:{lives}"
    
    def act(self, state, valid_actions, use_hmm_probs=True):
        """Choose action using epsilon-greedy policy."""
        state_key = self._get_state_key(state)
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Get Q-values for valid actions
        q_values = {action: self.q_table[state_key][action] 
                   for action in valid_actions}
        
        # Combine with HMM probabilities if available
        if use_hmm_probs and 'hmm_probs' in state:
            hmm_probs = state['hmm_probs']
            for action in valid_actions:
                action_idx = ord(action) - ord('a')
                hmm_score = hmm_probs[action_idx] * 100
                q_values[action] = 0.7 * q_values[action] + 0.3 * hmm_score
        
        # Choose best action
        return max(q_values, key=q_values.get)
    
    def learn(self, state, action, reward, next_state, done):
        """Update Q-value for state-action pair."""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_key][action]
        
        # Maximum Q-value for next state
        if done:
            max_next_q = 0
        else:
            next_state_dict = self.q_table[next_state_key]
            max_next_q = max(next_state_dict.values()) if next_state_dict else 0
        
        # Q-learning update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filename):
        """Save Q-table to file."""
        data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'gamma': self.gamma
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Q-Learning model saved to {filename}")
    
    def load(self, filename):
        """Load Q-table from file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        # Convert back to defaultdict
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state_key, actions in data['q_table'].items():
            for action, q_value in actions.items():
                self.q_table[state_key][action] = q_value
        
        self.epsilon = data.get('epsilon', self.epsilon_min)
        self.alpha = data.get('alpha', self.alpha)
        self.gamma = data.get('gamma', self.gamma)
        print(f"Q-Learning model loaded from {filename}")


# Test the agents
if __name__ == "__main__":
    print("Testing DQN Agent...")
    dqn_agent = HangmanDQNAgent()
    
    # Create dummy state
    test_state = {
        'masked_word': '_pp__',
        'guessed_letters': set(['e', 's']),
        'lives_remaining': 5,
        'hmm_probs': np.random.rand(26),
        'word_length': 5,
        'num_blanks': 3
    }
    
    valid_actions = ['a', 'b', 'c', 'd', 'l', 'o']
    action = dqn_agent.act(test_state, valid_actions)
    print(f"DQN chose action: {action}")
    
    print("\nTesting Q-Learning Agent...")
    ql_agent = SimpleQLearningAgent()
    action = ql_agent.act(test_state, valid_actions)
    print(f"Q-Learning chose action: {action}")
    
    print("\nAgents initialized successfully!")