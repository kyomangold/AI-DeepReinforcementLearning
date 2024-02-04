# Collaboration and Competition Project: Tennis Environment

## Introduction
This project involves training two agents to play tennis in a Unity ML-Agents environment. Each agent controls a racket with the goal of bouncing a ball over a net.

## Environment Details
- **State Space**: 8 variables representing the position and velocity of the ball and racket.
- **Action Space**: Continuous, with 2 actions corresponding to movement and jumping.
- **Reward**: +0.1 for hitting the ball over the net, -0.01 if the ball hits the ground or is out of bounds.
- **Goal**: Each agent aims to keep the ball in play.

## Solving the Environment
- The task is episodic.
- The environment is considered solved when the agents achieve an average score of +0.5 over 100 consecutive episodes.

## Getting Started
- Download the environment for your OS (Linux, Mac OSX, Windows 32-bit, Windows 64-bit).
- Place the downloaded file in the `p3_collab-compet/` folder and unzip it.

## Instructions
- Follow the instructions in `Tennis.ipynb` to train the agents.
- Ensure all dependencies are installed as per the `dependencies.md` file.

## Files in the Repository
- `Tennis.ipynb`: Jupyter notebook containing the training code.
- `agent.py`: Defines the DDPG agents used in training.
- `model.py`: Contains the neural network architectures for the Actor and Critic.

## Optional Challenge: Soccer Environment
- After completing this project, you can try the more challenging Soccer environment.
