# Hackman: An Intelligent Hangman Agent

This project is an intelligent Hangman assistant built for the "UE23CS352A: Machine Learning Hackathon". The goal is to move beyond simple human intuition and build an agent that learns the contextual patterns of English to solve Hangman puzzles with maximum efficiency.

The agent is designed to win games with the fewest possible mistakes by leveraging a hybrid system of probabilistic modeling and machine learning.

## 🤖 Project Architecture

The agent's design is a hybrid system that combines two key components:

1.  **Part 1: The "Oracle" (Hidden Markov Model)**
    * A **Hidden Markov Model (HMM)** is trained on the provided `corpus.txt` file (containing 50,000 English words).
    * Its purpose is to act as an "oracle," estimating the probability of each remaining letter appearing in each of the blank spots, given the current game state.

2.  **Part 2: The "Brain" (Reinforcement Learning Agent)**
    * A **Reinforcement Learning (RL) agent** (specifically, a Deep Q-Network) serves as the "brain".
    * It takes the probability distributions from the HMM as a key part of its state and decides the optimal letter to guess next.
    * The agent is trained in a custom Hangman environment with a reward function designed to maximize the win rate while heavily penalizing wrong and repeated guesses.

## 📂 File Structure

Here are the key files in this repository:

* `hangman.ipynb`: A Jupyter Notebook containing the complete, consolidated code for the project, from HMM training to RL evaluation.
* `hangman_hmm.py`: Python script for building and training the Hidden Markov Model.
* `hangman_environment.py`: Defines the custom Hangman game environment for the RL agent.
* `hangman_agent.py`: Defines the RL (DQN) agent architecture.
* `hangman_training.py`: The main script to run the RL agent's training loop.
* `hangman_evaluation.py`: Script to evaluate the trained agent and calculate the final score.
* `corpus.txt`: The 50,000-word dataset used for training the HMM.
* `test.txt`: The hidden test set used for evaluation.
* `hangman_dqn_final.pth`: The saved model weights for the trained DQN agent.
* `hangman_hmm_model.pkl`: The saved, trained HMM model.
* `/results/`: This folder contains evaluation results, including plots of the agent's learning (e.g., reward per episode).

## 🚀 How to Run

There are two ways to run this project:

### Option 1: The Jupyter Notebook (Recommended)

The `hangman.ipynb` file contains all code, explanations, and outputs in one place.

1.  Open the notebook in Jupyter Lab, Jupyter Notebook, or Google Colab.
2.  Install any required dependencies (e.g., `torch`, `numpy`).
3.  Run the cells sequentially from top to bottom to train the models and see the evaluation.

### Option 2: Using the Python Scripts

If you prefer to run the project from the command line, you must execute the scripts in the following order.

1.  **Train the HMM:**
    This script trains the HMM on `corpus.txt` and saves the model as `hangman_hmm_model.pkl`.
    ```bash
    python hangman_hmm.py
    ```

2.  **Train the RL Agent:**
    This script imports the environment and agent, loads the HMM, and trains the DQN, saving the final weights as `hangman_dqn_final.pth`.
    ```bash
    python hangman_training.py
    ```

3.  **Evaluate the Agent:**
    This script loads the trained HMM and DQN models, runs the evaluation against the `test.txt` set, and prints the final score.
    ```bash
    python hangman_evaluation.py
    ```

## 📊 Results

Evaluation plots (such as reward per episode) and final scores are saved in the `/results` folder. The final score is calculated based on the official hackathon formula:

`Final Score = (Success Rate * 2000) - (Total Wrong Guesses * 5) - (Total Repeated Guesses * 2)`