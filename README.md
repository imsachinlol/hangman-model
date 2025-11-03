🎯 Hackman: An Intelligent Hangman Agent

🧠 An intelligent Hangman assistant built for the “UE23CS352A: Machine Learning Hackathon”.
This project goes beyond simple human intuition — it learns contextual English patterns to solve Hangman puzzles with maximum efficiency using probabilistic modeling + reinforcement learning.

🏗️ Project Architecture

The system is a hybrid AI architecture combining two powerful components:

🧩 1. The “Oracle” — Hidden Markov Model (HMM)

A Hidden Markov Model is trained on the provided corpus.txt file containing 50,000 English words.

Acts as a probabilistic oracle that estimates the likelihood of each remaining letter appearing in the blank spaces, given the current game state.

⚙️ 2. The “Brain” — Reinforcement Learning Agent (DQN)

A Deep Q-Network (DQN) acts as the brain of the system.

Takes the HMM-generated probabilities as part of its state representation.

Decides the optimal next letter guess using experience-based learning.

Trained in a custom Hangman environment with a reward function that:

Maximizes win rate ✅

Penalizes wrong guesses ❌

Penalizes repeated guesses 🔁

📂 Repository Structure
File	Description
hangman.ipynb	Full Jupyter Notebook containing training + evaluation pipeline
hangman_hmm.py	Script for building and training the HMM
hangman_environment.py	Custom Hangman game environment
hangman_agent.py	Defines the RL (DQN) agent architecture
hangman_training.py	Main script to train the RL agent
hangman_evaluation.py	Script for evaluating the final trained agent
corpus.txt	50,000-word English corpus for HMM training
test.txt	Hidden test set for evaluation
hangman_dqn_final.pth	Trained DQN model weights
hangman_hmm_model.pkl	Trained HMM model
/results/	Folder containing evaluation plots and learning curves
🚀 How to Run

You can run this project in two ways:

🧑‍💻 Option 1: Run via Jupyter Notebook (Recommended)

Open hangman.ipynb in Jupyter Lab, Notebook, or Google Colab.

Install dependencies:

pip install torch numpy matplotlib tqdm


Run all cells sequentially from top to bottom to train and evaluate.

💻 Option 2: Run via Python Scripts
🏗️ Step 1 — Train the HMM
python hangman_hmm.py


👉 Trains the Hidden Markov Model on corpus.txt and saves it as hangman_hmm_model.pkl.

🧠 Step 2 — Train the RL Agent
python hangman_training.py


👉 Loads the trained HMM, initializes the Hangman environment, and trains the DQN agent.
Final model is saved as hangman_dqn_final.pth.

🧪 Step 3 — Evaluate the Agent
python hangman_evaluation.py


👉 Evaluates the final agent on test.txt and prints the final performance score.

📊 Results & Evaluation

All learning curves (reward per episode, loss trends, accuracy plots) are saved in the /results/ directory.

The official hackathon scoring formula is:

Final Score = (Success Rate * 2000) - (Total Wrong Guesses * 5) - (Total Repeated Guesses * 2)

🧩 Highlights

✅ Combines probabilistic modeling (HMM) with reinforcement learning (DQN)
✅ Learns contextual letter dependencies from a 50,000-word English corpus
✅ Custom Hangman simulation environment with well-designed reward shaping
✅ Optimized for minimal wrong guesses and maximal accuracy
✅ Fully modular — easy to extend with new agents or corpora

🏁 Future Work

🔹 Add transformer-based context modeling instead of HMM
🔹 Integrate curriculum learning for adaptive difficulty
🔹 Experiment with Policy Gradient / PPO for continuous improvement
🔹 Build a web interface to play against the AI

🧠 Authors & Acknowledgments

Developed for UE23CS352A: Machine Learning Hackathon
By: Team Hackman
Mentor: Department of Computer Science, PES University