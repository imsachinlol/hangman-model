"""
Training script for Hangman RL Agent.
Supports both DQN and Q-Learning agents.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
from hangman_hmm import HangmanHMM
from hangman_environment import HangmanEnvironment, load_corpus
from hangman_agent import HangmanDQNAgent, SimpleQLearningAgent

def smooth_curve(values, window=100):
    """Smooth curve using moving average."""
    if len(values) < window:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end = min(len(values), i + window // 2)
        smoothed.append(np.mean(values[start:end]))
    return smoothed

def plot_training_results(history, filename='training_results.png'):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = range(len(history['rewards']))
    
    # 1. Episode Rewards (smoothed)
    axes[0, 0].plot(episodes, history['rewards'], alpha=0.3, label='Raw')
    smoothed = smooth_curve(history['rewards'])
    axes[0, 0].plot(episodes, smoothed, label='Smoothed', linewidth=2)
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Win Rate
    win_rate_episodes = range(0, len(history['win_rates']) * 500, 500)
    axes[0, 1].plot(win_rate_episodes, history['win_rates'], 
                    marker='o', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Win Rate (%)')
    axes[0, 1].set_title('Win Rate Over Time (per 500 episodes)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 100])
    
    # 3. Wrong Guesses
    axes[1, 0].plot(win_rate_episodes, history['avg_wrong_guesses'], 
                    marker='s', color='orange', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Avg Wrong Guesses')
    axes[1, 0].set_title('Wrong Guesses Over Time (per 500 episodes)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Repeated Guesses
    axes[1, 1].plot(win_rate_episodes, history['avg_repeated_guesses'], 
                    marker='^', color='red', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Avg Repeated Guesses')
    axes[1, 1].set_title('Repeated Guesses Over Time (per 500 episodes)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nTraining plots saved to {filename}")
    plt.close()

def train_dqn(env, agent, num_episodes=5000, save_freq=1000):
    """Train DQN agent."""
    print("\n" + "="*60)
    print("Training DQN Agent")
    print("="*60)
    
    history = {
        'rewards': [],
        'losses': [],
        'win_rates': [],
        'avg_wrong_guesses': [],
        'avg_repeated_guesses': [],
        'epsilons': []
    }
    
    # Tracking variables
    episode_wins = []
    episode_wrong = []
    episode_repeated = []
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        done = False
        total_reward = 0
        losses = []
        
        while not done:
            # Choose action
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions, use_hmm_probs=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train on batch
            if len(agent.memory) > agent.batch_size:
                loss = agent.replay()
                losses.append(loss)
            
            state = next_state
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record episode stats
        history['rewards'].append(total_reward)
        history['losses'].extend(losses)
        history['epsilons'].append(agent.epsilon)
        
        episode_wins.append(1 if info.get('won') else 0)
        episode_wrong.append(env.wrong_guesses)
        episode_repeated.append(env.repeated_guesses)
        
        # Log progress every 500 episodes
        if (episode + 1) % 500 == 0:
            win_rate = np.mean(episode_wins[-500:]) * 100
            avg_wrong = np.mean(episode_wrong[-500:])
            avg_repeated = np.mean(episode_repeated[-500:])
            avg_reward = np.mean(history['rewards'][-500:])
            
            history['win_rates'].append(win_rate)
            history['avg_wrong_guesses'].append(avg_wrong)
            history['avg_repeated_guesses'].append(avg_repeated)
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Win rate (last 500): {win_rate:.2f}%")
            print(f"  Avg wrong guesses: {avg_wrong:.2f}")
            print(f"  Avg repeated guesses: {avg_repeated:.3f}")
            print(f"  Avg reward: {avg_reward:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint_file = f'hangman_dqn_checkpoint_{episode + 1}.pth'
            agent.save(checkpoint_file)
    
    return history

def train_qlearning(env, agent, num_episodes=10000):
    """Train Q-Learning agent."""
    print("\n" + "="*60)
    print("Training Q-Learning Agent")
    print("="*60)
    
    history = {
        'rewards': [],
        'win_rates': [],
        'avg_wrong_guesses': [],
        'avg_repeated_guesses': [],
        'epsilons': []
    }
    
    # Tracking variables
    episode_wins = []
    episode_wrong = []
    episode_repeated = []
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Choose action
            valid_actions = env.get_valid_actions()
            action = agent.act(state, valid_actions, use_hmm_probs=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            # Learn
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record episode stats
        history['rewards'].append(total_reward)
        history['epsilons'].append(agent.epsilon)
        
        episode_wins.append(1 if info.get('won') else 0)
        episode_wrong.append(env.wrong_guesses)
        episode_repeated.append(env.repeated_guesses)
        
        # Log progress every 500 episodes
        if (episode + 1) % 500 == 0:
            win_rate = np.mean(episode_wins[-500:]) * 100
            avg_wrong = np.mean(episode_wrong[-500:])
            avg_repeated = np.mean(episode_repeated[-500:])
            avg_reward = np.mean(history['rewards'][-500:])
            
            history['win_rates'].append(win_rate)
            history['avg_wrong_guesses'].append(avg_wrong)
            history['avg_repeated_guesses'].append(avg_repeated)
            
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Win rate (last 500): {win_rate:.2f}%")
            print(f"  Avg wrong guesses: {avg_wrong:.2f}")
            print(f"  Avg repeated guesses: {avg_repeated:.3f}")
            print(f"  Avg reward: {avg_reward:.2f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Q-table size: {len(agent.q_table)}")
    
    return history

def main():
    print("="*60)
    print("HANGMAN RL AGENT - TRAINING")
    print("="*60)
    
    # 1. Load HMM
    print("\n1. Loading HMM model...")
    try:
        hmm = HangmanHMM()
        hmm.load('hangman_hmm_model.pkl')
        print("   ✓ HMM loaded successfully")
    except FileNotFoundError:
        print("   ❌ HMM model not found!")
        print("   Please run 'python hangman_hmm.py' first to train the HMM.")
        return
    
    # 2. Load corpus
    print("\n2. Loading corpus...")
    try:
        words = load_corpus('corpus.txt')
        print(f"   ✓ Loaded {len(words)} words")
    except FileNotFoundError:
        print("   ❌ corpus.txt not found!")
        return
    
    # 3. Create environment
    print("\n3. Creating Hangman environment...")
    env = HangmanEnvironment(words, hmm, max_lives=6)
    print("   ✓ Environment created")
    
    # 4. Choose agent type
    print("\n4. Choose agent type:")
    print("   1. DQN (Deep Q-Network) - More powerful, slower")
    print("   2. Q-Learning (Table-based) - Simpler, faster")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    use_dqn = (choice == '1')
    
    # 5. Initialize agent
    if use_dqn:
        print("\n5. Initializing DQN agent...")
        agent = HangmanDQNAgent(
            state_size=55,
            action_size=26,
            learning_rate=0.0005,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.9995,
            memory_size=10000,
            batch_size=64
        )
        agent_name = "DQN"
        default_episodes = 5000
    else:
        print("\n5. Initializing Q-Learning agent...")
        agent = SimpleQLearningAgent(
            alpha=0.1,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.9995
        )
        agent_name = "Q-Learning"
        default_episodes = 10000
    
    print(f"   ✓ {agent_name} agent initialized")
    
    # 6. Get training parameters
    print(f"\n6. Training configuration:")
    print(f"   Recommended episodes for {agent_name}: {default_episodes}")
    
    while True:
        episodes_input = input(f"   Enter number of episodes (or press Enter for {default_episodes}): ").strip()
        if episodes_input == "":
            num_episodes = default_episodes
            break
        try:
            num_episodes = int(episodes_input)
            if num_episodes > 0:
                break
            print("   Please enter a positive number.")
        except ValueError:
            print("   Invalid input. Please enter a number.")
    
    print(f"\n   Training for {num_episodes} episodes...")
    print(f"   Estimated time: {num_episodes * 0.2 / 60:.1f} minutes")
    
    # 7. Train agent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if use_dqn:
        history = train_dqn(env, agent, num_episodes, save_freq=1000)
        model_file = f'hangman_dqn_final.pth'
    else:
        history = train_qlearning(env, agent, num_episodes)
        model_file = f'hangman_qlearning_final.pkl'
    
    # 8. Save final model
    print(f"\n8. Saving final model...")
    agent.save(model_file)
    
    # 9. Save training history
    history_file = f'training_history_{agent_name.lower()}_{timestamp}.json'
    with open(history_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_json = {k: [float(v) for v in vals] for k, vals in history.items()}
        json.dump(history_json, f, indent=2)
    print(f"   Training history saved to {history_file}")
    
    # 10. Plot results
    print("\n9. Generating training plots...")
    plot_file = f'training_results_{agent_name.lower()}_{timestamp}.png'
    plot_training_results(history, plot_file)
    
    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal Statistics:")
    if history['win_rates']:
        print(f"  Final win rate: {history['win_rates'][-1]:.2f}%")
        print(f"  Final avg wrong guesses: {history['avg_wrong_guesses'][-1]:.2f}")
        print(f"  Final avg repeated guesses: {history['avg_repeated_guesses'][-1]:.3f}")
    print(f"  Final epsilon: {history['epsilons'][-1]:.4f}")
    
    print(f"\nSaved files:")
    print(f"  - Model: {model_file}")
    print(f"  - History: {history_file}")
    print(f"  - Plots: {plot_file}")
    
    print(f"\nNext step: Evaluate your agent")
    print(f"  Run: python hangman_evaluation.py")

if __name__ == "__main__":
    main()