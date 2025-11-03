import numpy as np
import random
from collections import defaultdict

class HangmanEnvironment:
    """
    Hangman game environment for Reinforcement Learning.
    """
    
    def __init__(self, word_list, hmm_model, max_lives=6):
        """
        Args:
            word_list: List of words to use for games
            hmm_model: Trained HMM model for probability predictions
            max_lives: Maximum number of wrong guesses allowed
        """
        self.word_list = word_list
        self.hmm = hmm_model
        self.max_lives = max_lives
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        
        # Current game state
        self.target_word = None
        self.masked_word = None
        self.guessed_letters = None
        self.lives_remaining = None
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        
    def reset(self):
        """Start a new game with a random word."""
        self.target_word = random.choice(self.word_list).lower()
        self.masked_word = ['_'] * len(self.target_word)
        self.guessed_letters = set()
        self.lives_remaining = self.max_lives
        self.wrong_guesses = 0
        self.repeated_guesses = 0
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get current state representation.
        
        Returns dictionary containing:
        - masked_word: string representation
        - guessed_letters: set of guessed letters
        - lives_remaining: int
        - hmm_probs: probability distribution from HMM
        - word_length: int
        """
        masked_str = ''.join(self.masked_word)
        hmm_probs = self.hmm.get_letter_probabilities(masked_str, self.guessed_letters)
        
        state = {
            'masked_word': masked_str,
            'guessed_letters': self.guessed_letters.copy(),
            'lives_remaining': self.lives_remaining,
            'hmm_probs': hmm_probs,
            'word_length': len(self.target_word),
            'num_blanks': masked_str.count('_')
        }
        
        return state
    
    def get_state_vector(self):
        """
        Convert state to fixed-size vector for neural network input.
        
        Returns:
            numpy array suitable for DQN
        """
        state = self._get_state()
        
        # Components:
        # 1. Guessed letters (26 binary features)
        guessed_vec = np.zeros(26)
        for letter in state['guessed_letters']:
            idx = ord(letter) - ord('a')
            guessed_vec[idx] = 1
        
        # 2. Lives remaining (normalized)
        lives_vec = np.array([state['lives_remaining'] / self.max_lives])
        
        # 3. HMM probabilities (26 features)
        hmm_vec = state['hmm_probs']
        
        # 4. Word length (normalized)
        length_vec = np.array([state['word_length'] / 20.0])  # Assuming max length ~20
        
        # 5. Number of blanks (normalized)
        blanks_vec = np.array([state['num_blanks'] / state['word_length']])
        
        # Concatenate all features
        state_vector = np.concatenate([
            guessed_vec,    # 26
            lives_vec,      # 1
            hmm_vec,        # 26
            length_vec,     # 1
            blanks_vec      # 1
        ])  # Total: 55 features
        
        return state_vector
    
    def step(self, action):
        """
        Take an action (guess a letter).
        
        Args:
            action: letter to guess (string or int index)
            
        Returns:
            next_state, reward, done, info
        """
        # Convert action to letter if it's an index
        if isinstance(action, int):
            letter = chr(action + ord('a'))
        else:
            letter = action.lower()
        
        # Check if letter was already guessed
        if letter in self.guessed_letters:
            self.repeated_guesses += 1
            reward = -50  # Heavy penalty for repeated guess
            done = False
            info = {'repeated': True, 'wrong': False}
            return self._get_state(), reward, done, info
        
        # Add to guessed letters
        self.guessed_letters.add(letter)
        
        # Check if letter is in the word
        if letter in self.target_word:
            # Correct guess - reveal letters
            count = 0
            for i, char in enumerate(self.target_word):
                if char == letter:
                    self.masked_word[i] = letter
                    count += 1
            
            # Check if word is complete
            if '_' not in self.masked_word:
                reward = 100 + (self.lives_remaining * 10)  # Win bonus + life bonus
                done = True
                info = {'won': True, 'wrong': False, 'letters_revealed': count}
            else:
                reward = 10 * count  # Reward proportional to letters revealed
                done = False
                info = {'won': False, 'wrong': False, 'letters_revealed': count}
        else:
            # Wrong guess
            self.lives_remaining -= 1
            self.wrong_guesses += 1
            
            if self.lives_remaining <= 0:
                reward = -100  # Loss penalty
                done = True
                info = {'won': False, 'wrong': True, 'lost': True}
            else:
                reward = -20  # Wrong guess penalty
                done = False
                info = {'won': False, 'wrong': True, 'lost': False}
        
        next_state = self._get_state()
        return next_state, reward, done, info
    
    def get_valid_actions(self):
        """Get list of letters that haven't been guessed yet."""
        valid = []
        for letter in self.alphabet:
            if letter not in self.guessed_letters:
                valid.append(letter)
        return valid
    
    def get_valid_action_indices(self):
        """Get indices of valid actions (for neural network output)."""
        valid = []
        for i, letter in enumerate(self.alphabet):
            if letter not in self.guessed_letters:
                valid.append(i)
        return valid
    
    def render(self):
        """Print current game state."""
        print(f"Word: {' '.join(self.masked_word)}")
        print(f"Guessed: {sorted(self.guessed_letters)}")
        print(f"Lives: {self.lives_remaining}/{self.max_lives}")
        print(f"Wrong guesses: {self.wrong_guesses}")


def load_corpus(filename):
    """Load word list from corpus file."""
    with open(filename, 'r') as f:
        words = [line.strip().lower() for line in f if line.strip()]
    
    # Filter only alphabetic words
    words = [w for w in words if w.isalpha() and len(w) >= 2]
    return words


# Test the environment
if __name__ == "__main__":
    from hangman_hmm import HangmanHMM
    
    print("Loading HMM model...")
    hmm = HangmanHMM()
    hmm.load('hangman_hmm_model.pkl')
    
    print("Loading corpus...")
    words = load_corpus('corpus.txt')
    print(f"Loaded {len(words)} words")
    
    print("\nTesting environment...")
    env = HangmanEnvironment(words, hmm)
    
    # Play a test game
    state = env.reset()
    print(f"\nTarget word (hidden): {env.target_word}")
    env.render()
    
    done = False
    while not done:
        # Get HMM recommendation
        hmm_probs = state['hmm_probs']
        valid_actions = env.get_valid_actions()
        
        # Choose letter with highest probability
        best_idx = np.argmax(hmm_probs)
        guess = chr(best_idx + ord('a'))
        
        # Make sure it's valid
        if guess not in valid_actions:
            guess = random.choice(valid_actions)
        
        print(f"\nGuessing: {guess}")
        state, reward, done, info = env.step(guess)
        print(f"Reward: {reward}, Info: {info}")
        env.render()
    
    if info.get('won'):
        print("\n🎉 Won the game!")
    else:
        print(f"\n❌ Lost! Word was: {env.target_word}")