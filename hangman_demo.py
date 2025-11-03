"""
Interactive demo script for Hangman AI agent.
Allows you to watch the agent play or play against it.
"""

import time
from hangman_hmm import HangmanHMM
from hangman_environment import HangmanEnvironment, load_corpus
from hangman_agent import HangmanDQNAgent, SimpleQLearningAgent

def print_game_state(env, guess=None, result=None):
    """Pretty print current game state."""
    print("\n" + "="*50)
    print(f"Word: {' '.join(env.masked_word)}")
    print(f"Lives: {'❤️ ' * env.lives_remaining}{'💀 ' * (env.max_lives - env.lives_remaining)}")
    print(f"Guessed: {', '.join(sorted(env.guessed_letters))}")
    
    if guess:
        if result:
            status = "✓ Correct!" if not result.get('wrong') else "✗ Wrong!"
            print(f"\nGuessed '{guess.upper()}': {status}")
            if result.get('letters_revealed'):
                print(f"Revealed {result['letters_revealed']} letter(s)!")
    
    print("="*50)

def watch_agent_play(env, agent, num_games=5, delay=0.5):
    """Watch the agent play Hangman games."""
    print("\n" + "🎮" * 25)
    print("WATCHING AI AGENT PLAY HANGMAN")
    print("🎮" * 25)
    
    for game_num in range(num_games):
        print(f"\n\n{'='*60}")
        print(f"GAME {game_num + 1}/{num_games}")
        print(f"{'='*60}")
        
        state = env.reset()
        print(f"\nTarget word has {len(env.target_word)} letters")
        
        done = False
        turn = 0
        
        while not done:
            turn += 1
            print(f"\n--- Turn {turn} ---")
            print_game_state(env)
            
            # Get agent's move
            valid_actions = env.get_valid_actions()
            guess = agent.act(state, valid_actions, use_hmm_probs=True)
            
            # Get HMM probabilities for explanation
            hmm_probs = state['hmm_probs']
            top_5_indices = sorted(range(26), key=lambda i: hmm_probs[i], reverse=True)[:5]
            top_5_letters = [(chr(i + ord('a')), hmm_probs[i]) for i in top_5_indices if chr(i + ord('a')) in valid_actions]
            
            print(f"\nAgent thinking...")
            print(f"Top 5 HMM suggestions: {[(l, f'{p:.3f}') for l, p in top_5_letters[:5]]}")
            print(f"Agent decides: '{guess.upper()}'")
            
            time.sleep(delay)
            
            # Make the move
            state, reward, done, info = env.step(guess)
            print_game_state(env, guess, info)
            
            if not done:
                time.sleep(delay)
        
        # Game over
        print("\n" + "🎊" * 30)
        if info.get('won'):
            print(f"🎉 AGENT WON! Word was: {env.target_word.upper()}")
            print(f"Solved in {turn} guesses with {env.wrong_guesses} wrong guesses")
        else:
            print(f"💀 AGENT LOST! Word was: {env.target_word.upper()}")
            print(f"Made {env.wrong_guesses} wrong guesses")
        print("🎊" * 30)
        
        if game_num < num_games - 1:
            input("\nPress Enter for next game...")

def human_vs_ai(env, agent):
    """Play Hangman with AI assistance."""
    print("\n" + "🎮" * 25)
    print("HANGMAN WITH AI ASSISTANT")
    print("🎮" * 25)
    print("\nThe AI will suggest letters, but you make the final decision!")
    
    while True:
        state = env.reset()
        print(f"\n\n{'='*60}")
        print("NEW GAME")
        print(f"{'='*60}")
        print(f"Word has {len(env.target_word)} letters")
        
        done = False
        turn = 0
        
        while not done:
            turn += 1
            print(f"\n--- Turn {turn} ---")
            print_game_state(env)
            
            # Get AI suggestion
            valid_actions = env.get_valid_actions()
            ai_suggestion = agent.act(state, valid_actions, use_hmm_probs=True)
            
            # Get top suggestions
            hmm_probs = state['hmm_probs']
            top_5_indices = sorted(range(26), key=lambda i: hmm_probs[i], reverse=True)[:5]
            top_5_letters = [(chr(i + ord('a')), hmm_probs[i]) for i in top_5_indices if chr(i + ord('a')) in valid_actions]
            
            print(f"\n🤖 AI Suggestions:")
            for i, (letter, prob) in enumerate(top_5_letters[:5], 1):
                marker = "⭐" if letter == ai_suggestion else "  "
                print(f"{marker} {i}. {letter.upper()} (confidence: {prob:.1%})")
            
            # Get human input
            while True:
                guess = input(f"\nYour guess (or 'a' for AI choice '{ai_suggestion.upper()}'): ").lower().strip()
                
                if guess == 'a':
                    guess = ai_suggestion
                    print(f"Using AI suggestion: {guess.upper()}")
                    break
                elif len(guess) == 1 and guess.isalpha():
                    if guess in env.guessed_letters:
                        print(f"Already guessed '{guess.upper()}'! Try again.")
                    else:
                        break
                else:
                    print("Please enter a single letter or 'a' for AI suggestion.")
            
            # Make the move
            state, reward, done, info = env.step(guess)
            print_game_state(env, guess, info)
            
            if not done:
                time.sleep(0.5)
        
        # Game over
        print("\n" + "🎊" * 30)
        if info.get('won'):
            print(f"🎉 YOU WON! Word was: {env.target_word.upper()}")
            print(f"Solved in {turn} guesses with {env.wrong_guesses} wrong guesses")
        else:
            print(f"💀 GAME OVER! Word was: {env.target_word.upper()}")
        print("🎊" * 30)
        
        play_again = input("\nPlay again? (y/n): ").lower().strip()
        if play_again != 'y':
            break
    
    print("\nThanks for playing! 👋")

def quick_benchmark(env, agent, num_games=100):
    """Quick performance benchmark."""
    print("\n" + "📊" * 25)
    print(f"QUICK BENCHMARK - {num_games} GAMES")
    print("📊" * 25)
    
    wins = 0
    total_wrong = 0
    total_repeated = 0
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions()
            guess = agent.act(state, valid_actions, use_hmm_probs=True)
            state, reward, done, info = env.step(guess)
        
        if info.get('won'):
            wins += 1
        total_wrong += env.wrong_guesses
        total_repeated += env.repeated_guesses
    
    print(f"\nResults:")
    print(f"  Games played: {num_games}")
    print(f"  Wins: {wins} ({wins/num_games*100:.1f}%)")
    print(f"  Avg wrong guesses: {total_wrong/num_games:.2f}")
    print(f"  Avg repeated guesses: {total_repeated/num_games:.2f}")
    
    final_score = (wins/num_games * 2000) - (total_wrong * 5) - (total_repeated * 2)
    print(f"\n  Estimated Final Score: {final_score:.0f}")

def main():
    """Main demo script."""
    print("="*60)
    print("HANGMAN AI - INTERACTIVE DEMO")
    print("="*60)
    
    # Load HMM
    print("\nLoading HMM model...")
    try:
        hmm = HangmanHMM()
        hmm.load('hangman_hmm_model.pkl')
    except FileNotFoundError:
        print("❌ HMM model not found! Please run 'python hangman_hmm.py' first.")
        return
    
    # Load corpus
    print("Loading corpus...")
    try:
        words = load_corpus('corpus.txt')
        print(f"✓ Loaded {len(words)} words")
    except FileNotFoundError:
        print("❌ corpus.txt not found!")
        return
    
    # Create environment
    env = HangmanEnvironment(words, hmm, max_lives=6)
    
    # Load agent
    print("\nChoose agent type:")
    print("  1. DQN (Deep Q-Network)")
    print("  2. Q-Learning (Table-based)")
    
    agent_choice = input("Enter choice (1 or 2): ").strip()
    
    try:
        if agent_choice == '1':
            print("\nLoading DQN agent...")
            agent = HangmanDQNAgent()
            agent.load('hangman_dqn_final.pth')
            agent.epsilon = 0.0  # Greedy evaluation
        else:
            print("\nLoading Q-Learning agent...")
            agent = SimpleQLearningAgent()
            agent.load('hangman_qlearning_final.pkl')
            agent.epsilon = 0.0  # Greedy evaluation
        print("✓ Agent loaded successfully!")
    except FileNotFoundError:
        print("❌ Trained agent not found! Please train an agent first.")
        print("   Run: python hangman_training.py")
        return
    
    # Main menu
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Watch AI play")
        print("2. Play with AI assistance")
        print("3. Quick benchmark")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            num_games = int(input("How many games to watch? (1-10): ") or "3")
            delay = float(input("Delay between moves in seconds (0-2): ") or "0.5")
            watch_agent_play(env, agent, num_games, delay)
        
        elif choice == '2':
            human_vs_ai(env, agent)
        
        elif choice == '3':
            num_games = int(input("How many games for benchmark? (10-1000): ") or "100")
            quick_benchmark(env, agent, num_games)
        
        elif choice == '4':
            print("\nGoodbye! 👋")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()