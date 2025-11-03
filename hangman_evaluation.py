"""
Evaluation script for Hangman RL Agent.
Tests agent performance and generates detailed analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
from collections import defaultdict
from hangman_hmm import HangmanHMM
from hangman_environment import HangmanEnvironment, load_corpus
from hangman_agent import HangmanDQNAgent, SimpleQLearningAgent

def evaluate_agent(env, agent, num_games=2000, verbose=False):
    """
    Evaluate agent on test games.
    
    Returns:
        dict: Detailed evaluation metrics
    """
    results = {
        'games_played': 0,
        'games_won': 0,
        'games_lost': 0,
        'total_wrong_guesses': 0,
        'total_repeated_guesses': 0,
        'wrong_guesses_per_game': [],
        'repeated_guesses_per_game': [],
        'game_lengths': [],
        'difficult_words': [],  # Words that caused losses
        'perfect_games': 0,  # Won with 0 wrong guesses
    }
    
    for game_num in tqdm(range(num_games), desc="Evaluating"):
        state = env.reset()
        done = False
        num_turns = 0
        
        if verbose and game_num < 5:
            print(f"\n=== Game {game_num + 1} ===")
            print(f"Target word: {env.target_word}")
        
        while not done:
            num_turns += 1
            
            # Get action from agent (greedy, no exploration)
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions, use_hmm_probs=True)
            
            # Take action
            state, reward, done, info = env.step(action)
            
            if verbose and game_num < 5:
                print(f"Turn {num_turns}: Guessed '{action}' -> {env.masked_word}")
        
        # Record results
        results['games_played'] += 1
        
        if info.get('won'):
            results['games_won'] += 1
            if env.wrong_guesses == 0:
                results['perfect_games'] += 1
        else:
            results['games_lost'] += 1
            results['difficult_words'].append({
                'word': env.target_word,
                'wrong_guesses': env.wrong_guesses,
                'repeated_guesses': env.repeated_guesses,
                'turns': num_turns
            })
        
        results['total_wrong_guesses'] += env.wrong_guesses
        results['total_repeated_guesses'] += env.repeated_guesses
        results['wrong_guesses_per_game'].append(env.wrong_guesses)
        results['repeated_guesses_per_game'].append(env.repeated_guesses)
        results['game_lengths'].append(num_turns)
        
        if verbose and game_num < 5:
            if info.get('won'):
                print(f"✓ Won! ({env.wrong_guesses} wrong guesses)")
            else:
                print(f"✗ Lost! Word was: {env.target_word}")
    
    return results

def plot_evaluation_results(results, filename='evaluation_results.png'):
    """Generate evaluation plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Wrong Guesses Distribution
    axes[0, 0].hist(results['wrong_guesses_per_game'], bins=range(0, 8), 
                    edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Wrong Guesses per Game')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Wrong Guesses')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Repeated Guesses Distribution
    max_repeated = max(results['repeated_guesses_per_game']) + 1
    axes[0, 1].hist(results['repeated_guesses_per_game'], 
                    bins=range(0, max_repeated + 1), 
                    edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Repeated Guesses per Game')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Repeated Guesses')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Win/Loss Pie Chart
    labels = ['Won', 'Lost']
    sizes = [results['games_won'], results['games_lost']]
    colors = ['#90EE90', '#FFB6C6']
    explode = (0.05, 0)
    
    axes[1, 0].pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
    axes[1, 0].set_title('Win/Loss Ratio')
    
    # 4. Performance Metrics Summary
    axes[1, 1].axis('off')
    
    success_rate = results['games_won'] / results['games_played'] * 100
    avg_wrong = results['total_wrong_guesses'] / results['games_played']
    avg_repeated = results['total_repeated_guesses'] / results['games_played']
    
    summary_text = f"""
    Performance Summary
    ═══════════════════════════════════
    
    Games Played:           {results['games_played']:,}
    Games Won:              {results['games_won']:,}
    Games Lost:             {results['games_lost']:,}
    
    Success Rate:           {success_rate:.2f}%
    Perfect Games:          {results['perfect_games']} ({results['perfect_games']/results['games_played']*100:.1f}%)
    
    Total Wrong Guesses:    {results['total_wrong_guesses']:,}
    Avg Wrong/Game:         {avg_wrong:.3f}
    
    Total Repeated Guesses: {results['total_repeated_guesses']:,}
    Avg Repeated/Game:      {avg_repeated:.3f}
    
    Avg Game Length:        {np.mean(results['game_lengths']):.1f} turns
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nEvaluation plots saved to {filename}")
    plt.close()

def analyze_difficult_words(results, top_n=20):
    """Analyze most difficult words."""
    if not results['difficult_words']:
        print("\nNo failed games to analyze!")
        return
    
    # Sort by wrong guesses
    sorted_words = sorted(results['difficult_words'], 
                         key=lambda x: x['wrong_guesses'], 
                         reverse=True)
    
    print("\n" + "="*60)
    print(f"TOP {top_n} MOST DIFFICULT WORDS")
    print("="*60)
    
    for i, word_info in enumerate(sorted_words[:top_n], 1):
        print(f"\n{i}. '{word_info['word'].upper()}'")
        print(f"   Wrong guesses: {word_info['wrong_guesses']}")
        print(f"   Repeated guesses: {word_info['repeated_guesses']}")
        print(f"   Total turns: {word_info['turns']}")
    
    # Word length analysis
    word_lengths = defaultdict(list)
    for word_info in results['difficult_words']:
        length = len(word_info['word'])
        word_lengths[length].append(word_info['wrong_guesses'])
    
    print("\n" + "="*60)
    print("DIFFICULTY BY WORD LENGTH")
    print("="*60)
    
    for length in sorted(word_lengths.keys()):
        avg_wrong = np.mean(word_lengths[length])
        count = len(word_lengths[length])
        print(f"Length {length:2d}: {count:3d} failed games, "
              f"avg {avg_wrong:.2f} wrong guesses")

def calculate_final_score(results):
    """Calculate final score according to competition formula."""
    success_rate = results['games_won'] / results['games_played']
    total_wrong = results['total_wrong_guesses']
    total_repeated = results['total_repeated_guesses']
    
    score = (success_rate * 2000) - (total_wrong * 5) - (total_repeated * 2)
    
    return score

def print_evaluation_summary(results):
    """Print detailed evaluation summary."""
    success_rate = results['games_won'] / results['games_played'] * 100
    avg_wrong = results['total_wrong_guesses'] / results['games_played']
    avg_repeated = results['total_repeated_guesses'] / results['games_played']
    final_score = calculate_final_score(results)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nGames Played: {results['games_played']}")
    print(f"Games Won: {results['games_won']}")
    print(f"Games Lost: {results['games_lost']}")
    
    print(f"\nSuccess Rate: {success_rate:.2f}%")
    print(f"Perfect Games (0 wrong): {results['perfect_games']} "
          f"({results['perfect_games']/results['games_played']*100:.1f}%)")
    
    print(f"\nTotal Wrong Guesses: {results['total_wrong_guesses']}")
    print(f"Avg Wrong Guesses per Game: {avg_wrong:.3f}")
    
    print(f"\nTotal Repeated Guesses: {results['total_repeated_guesses']}")
    print(f"Avg Repeated Guesses per Game: {avg_repeated:.3f}")
    
    print(f"\nAvg Game Length: {np.mean(results['game_lengths']):.1f} turns")
    
    print("\n" + "-"*60)
    print("FINAL SCORE CALCULATION")
    print("-"*60)
    print(f"Success Rate × 2000 = {success_rate/100 * 2000:.2f}")
    print(f"Wrong Guesses × 5 = {results['total_wrong_guesses'] * 5:.2f}")
    print(f"Repeated Guesses × 2 = {results['total_repeated_guesses'] * 2:.2f}")
    print("")
    print(f"FINAL SCORE: {final_score:.2f}")
    print("="*60)

def main():
    print("="*60)
    print("HANGMAN RL AGENT - EVALUATION")
    print("="*60)
    
    # 1. Load HMM
    print("\n1. Loading HMM model...")
    try:
        hmm = HangmanHMM()
        hmm.load('hangman_hmm_model.pkl')
        print("   ✓ HMM loaded")
    except FileNotFoundError:
        print("   ❌ HMM model not found!")
        return
    
    # 2. Load corpus
    print("\n2. Loading corpus...")
    try:
        all_words = load_corpus('corpus.txt')
        # Use last 20% for testing
        test_start = int(len(all_words) * 0.8)
        test_words = all_words[test_start:]
        print(f"   ✓ Loaded {len(test_words)} test words")
    except FileNotFoundError:
        print("   ❌ corpus.txt not found!")
        return
    
    # 3. Create environment
    print("\n3. Creating evaluation environment...")
    env = HangmanEnvironment(test_words, hmm, max_lives=6)
    print("   ✓ Environment created")
    
    # 4. Choose agent type
    print("\n4. Choose agent to evaluate:")
    print("   1. DQN")
    print("   2. Q-Learning")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Invalid choice.")
    
    use_dqn = (choice == '1')
    
    # 5. Load agent
    print(f"\n5. Loading {'DQN' if use_dqn else 'Q-Learning'} agent...")
    
    if use_dqn:
        agent = HangmanDQNAgent()
        default_file = 'hangman_dqn_final.pth'
    else:
        agent = SimpleQLearningAgent()
        default_file = 'hangman_qlearning_final.pkl'
    
    model_file = input(f"   Model file (or press Enter for '{default_file}'): ").strip()
    if not model_file:
        model_file = default_file
    
    try:
        agent.load(model_file)
        agent.epsilon = 0.0  # Greedy evaluation, no exploration
        print(f"   ✓ Agent loaded (epsilon set to 0 for evaluation)")
    except FileNotFoundError:
        print(f"   ❌ Model file '{model_file}' not found!")
        return
    
    # 6. Evaluation parameters
    print("\n6. Evaluation settings:")
    num_games_input = input("   Number of games (press Enter for 2000): ").strip()
    num_games = int(num_games_input) if num_games_input else 2000
    
    verbose_input = input("   Verbose output for first 5 games? (y/n, default n): ").strip().lower()
    verbose = (verbose_input == 'y')
    
    # 7. Run evaluation
    print(f"\n7. Evaluating agent on {num_games} games...")
    results = evaluate_agent(env, agent, num_games, verbose)
    
    # 8. Print summary
    print_evaluation_summary(results)
    
    # 9. Analyze difficult words
    if results['games_lost'] > 0:
        analyze_difficult_words(results, top_n=20)
    
    # 10. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_name = "dqn" if use_dqn else "qlearning"
    
    results_file = f'evaluation_results_{agent_name}_{timestamp}.json'
    
    # Convert results to JSON-serializable format
    results_json = {
        'games_played': results['games_played'],
        'games_won': results['games_won'],
        'games_lost': results['games_lost'],
        'success_rate': results['games_won'] / results['games_played'] * 100,
        'total_wrong_guesses': results['total_wrong_guesses'],
        'total_repeated_guesses': results['total_repeated_guesses'],
        'avg_wrong_guesses': results['total_wrong_guesses'] / results['games_played'],
        'avg_repeated_guesses': results['total_repeated_guesses'] / results['games_played'],
        'perfect_games': results['perfect_games'],
        'final_score': float(calculate_final_score(results)),
        'difficult_words': results['difficult_words'][:50]  # Save top 50
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # 11. Generate plots
    print("\n11. Generating evaluation plots...")
    plot_file = f'evaluation_results_{agent_name}_{timestamp}.png'
    plot_evaluation_results(results, plot_file)
    
    # Final message
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    
    final_score = calculate_final_score(results)
    
    if final_score > 1500:
        grade = "A+"
    elif final_score > 1000:
        grade = "A"
    elif final_score > 500:
        grade = "B"
    elif final_score > 0:
        grade = "C"
    else:
        grade = "D/F"
    
    print(f"\nYour estimated grade: {grade}")
    print(f"Final Score: {final_score:.2f}")
    
    print(f"\nGenerated files:")
    print(f"  - Results: {results_file}")
    print(f"  - Plots: {plot_file}")

if __name__ == "__main__":
    main()