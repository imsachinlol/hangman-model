import numpy as np
import pickle
from collections import defaultdict, Counter
import re

class HangmanHMM:
    """
    Hidden Markov Model for Hangman letter prediction.
    Trains separate models for each word length.
    """
    
    def __init__(self):
        self.models = {}  # Dictionary of models by word length
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
        self.letter_to_idx = {letter: idx for idx, letter in enumerate(self.alphabet)}
        self.idx_to_letter = {idx: letter for idx, letter in enumerate(self.alphabet)}
        
    def train(self, corpus_file):
        """Train HMM on corpus file."""
        print("Loading corpus...")
        with open(corpus_file, 'r') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        
        # Filter only alphabetic words
        words = [w for w in words if w.isalpha()]
        
        print(f"Loaded {len(words)} words")
        
        # Group words by length
        words_by_length = defaultdict(list)
        for word in words:
            words_by_length[len(word)].append(word)
        
        print(f"Training models for {len(words_by_length)} different word lengths...")
        
        # Train a model for each word length
        for length, word_list in words_by_length.items():
            if length < 2:  # Skip very short words
                continue
            print(f"Training length {length}: {len(word_list)} words")
            self.models[length] = self._train_length_model(word_list, length)
        
        print("Training complete!")
        
    def _train_length_model(self, words, length):
        """Train HMM for specific word length."""
        model = {
            'length': length,
            'position_probs': np.zeros((length, 26)),  # P(letter | position)
            'transition_probs': np.zeros((26, 26)),     # P(letter_t | letter_t-1)
            'initial_probs': np.zeros(26),              # P(first letter)
            'word_count': len(words)
        }
        
        # Count letter occurrences at each position
        position_counts = np.zeros((length, 26))
        
        for word in words:
            # Initial letter
            if len(word) > 0:
                model['initial_probs'][self.letter_to_idx[word[0]]] += 1
            
            # Position-specific counts
            for pos, letter in enumerate(word):
                if letter in self.letter_to_idx:
                    position_counts[pos][self.letter_to_idx[letter]] += 1
            
            # Transition counts (bigrams)
            for i in range(len(word) - 1):
                curr_letter = word[i]
                next_letter = word[i + 1]
                if curr_letter in self.letter_to_idx and next_letter in self.letter_to_idx:
                    curr_idx = self.letter_to_idx[curr_letter]
                    next_idx = self.letter_to_idx[next_letter]
                    model['transition_probs'][curr_idx][next_idx] += 1
        
        # Normalize to get probabilities
        # Position probabilities with smoothing
        for pos in range(length):
            total = position_counts[pos].sum() + 26  # Laplace smoothing
            model['position_probs'][pos] = (position_counts[pos] + 1) / total
        
        # Initial probabilities with smoothing
        total = model['initial_probs'].sum() + 26
        model['initial_probs'] = (model['initial_probs'] + 1) / total
        
        # Transition probabilities with smoothing
        for i in range(26):
            total = model['transition_probs'][i].sum() + 26
            model['transition_probs'][i] = (model['transition_probs'][i] + 1) / total
        
        return model
    
    def get_letter_probabilities(self, masked_word, guessed_letters):
        """
        Get probability distribution over letters given current game state.
        
        Args:
            masked_word: str, e.g., "_ppl_"
            guessed_letters: set of already guessed letters
            
        Returns:
            numpy array of probabilities for each letter (26 dimensions)
        """
        length = len(masked_word)
        
        if length not in self.models:
            # Fallback to general letter frequency
            return self._get_frequency_based_probs(guessed_letters)
        
        model = self.models[length]
        letter_scores = np.zeros(26)
        
        # Aggregate probabilities from each blank position
        for pos, char in enumerate(masked_word):
            if char == '_':
                # Add position-based probability
                letter_scores += model['position_probs'][pos]
                
                # Consider transitions from known adjacent letters
                if pos > 0 and masked_word[pos - 1] != '_':
                    prev_letter = masked_word[pos - 1]
                    prev_idx = self.letter_to_idx[prev_letter]
                    letter_scores += model['transition_probs'][prev_idx] * 0.5
                
                if pos < length - 1 and masked_word[pos + 1] != '_':
                    # Use reverse transition as approximation
                    next_letter = masked_word[pos + 1]
                    next_idx = self.letter_to_idx[next_letter]
                    for i in range(26):
                        letter_scores[i] += model['transition_probs'][i][next_idx] * 0.3
        
        # Zero out already guessed letters
        for letter in guessed_letters:
            if letter in self.letter_to_idx:
                letter_scores[self.letter_to_idx[letter]] = 0
        
        # Normalize
        if letter_scores.sum() > 0:
            letter_scores = letter_scores / letter_scores.sum()
        else:
            # Uniform distribution over unguessed letters
            letter_scores = np.ones(26)
            for letter in guessed_letters:
                if letter in self.letter_to_idx:
                    letter_scores[self.letter_to_idx[letter]] = 0
            if letter_scores.sum() > 0:
                letter_scores = letter_scores / letter_scores.sum()
        
        return letter_scores
    
    def _get_frequency_based_probs(self, guessed_letters):
        """Fallback: general English letter frequency."""
        freq = np.array([
            0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015,
            0.06094, 0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749,
            0.07507, 0.01929, 0.00095, 0.05987, 0.06327, 0.09056, 0.02758,
            0.00978, 0.02360, 0.00150, 0.01974, 0.00074
        ])
        
        for letter in guessed_letters:
            if letter in self.letter_to_idx:
                freq[self.letter_to_idx[letter]] = 0
        
        if freq.sum() > 0:
            freq = freq / freq.sum()
        
        return freq
    
    def save(self, filename):
        """Save trained model to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.models, f)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load trained model from file."""
        with open(filename, 'rb') as f:
            self.models = pickle.load(f)
        print(f"Model loaded from {filename}")


# Training script
if __name__ == "__main__":
    print("=" * 50)
    print("Hangman HMM Training")
    print("=" * 50)
    
    # Initialize and train HMM
    hmm = HangmanHMM()
    hmm.train('corpus.txt')
    
    # Save the trained model
    hmm.save('hangman_hmm_model.pkl')
    
    # Test the model
    print("\n" + "=" * 50)
    print("Testing HMM")
    print("=" * 50)
    
    test_cases = [
        ("_ppl_", set(['e', 's'])),
        ("h_ll_", set(['e'])),
        ("_____", set()),
        ("pro___m", set(['a', 'e', 'i']))
    ]
    
    for masked_word, guessed in test_cases:
        probs = hmm.get_letter_probabilities(masked_word, guessed)
        
        # Get top 5 letters
        top_indices = np.argsort(probs)[-5:][::-1]
        top_letters = [(hmm.idx_to_letter[idx], probs[idx]) for idx in top_indices]
        
        print(f"\nMasked word: {masked_word}")
        print(f"Guessed: {guessed}")
        print(f"Top 5 predictions: {[(l, f'{p:.4f}') for l, p in top_letters]}")