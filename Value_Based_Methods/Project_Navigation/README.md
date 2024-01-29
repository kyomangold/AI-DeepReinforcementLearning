[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

## Introduction

In this project, I have trained a deep reinforcement learning agent to collect (yellow) bananas in the Unity Banana Collector environment.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

## Implementation

This deep reinforcement learning agent is implemented using PyTorch and is designed for environments with continuous or discrete state and action spaces. The core components of this implementation include a neural network model for the agent's policy and methods for interacting with and learning from the environment.

### Neural Network Architecture (model.py)

QNetwork Class: Represents the Actor (Policy) Model.
Layers: Comprises fully connected layers with 256 neurons each, designed to map states to action values.
Activation: Uses ReLU activation for hidden layers.

### Agent Mechanics (dqn_agent.py)

Agent Class: Manages interactions with the environment and learning process.
Replay Buffer: Utilizes a replay buffer to store and sample experiences, aiding in stable learning.
Learning Strategy: Employs the Deep Q-Network (DQN) approach, updating the agent's knowledge based on sampled experiences from the replay buffer.
Target Network: Includes a mechanism for soft updates to the target network, ensuring gradual learning.
Device Adaptation: The code is adapted to leverage GPU acceleration, including compatibility with Apple's M1 chip using MPS when available, falling back to CPU otherwise.
This agent's implementation is optimized for robust performance and efficiency, capable of handling complex environments and tasks in reinforcement learning.

### Hyperparameters

In the Deep Reinforcement Learning (DRL) agent, several hyperparameters have been chosen to optimize the learning process:

1. **BUFFER_SIZE (100,000)**: This defines the size of the replay buffer, a data structure used to store and recall experience tuples (state, action, reward, next state). A larger buffer allows the agent to remember and learn from more experiences, potentially improving learning quality.

2. **BATCH_SIZE (64)**: This is the number of experience tuples that the agent samples from the replay buffer for learning. A batch size of 64 strikes a balance between computational efficiency and the quality of learning from diverse experiences.

3. **GAMMA (0.99)**: The discount factor determines the importance of future rewards. A value of 0.99 places high importance on future rewards, encouraging the agent to plan over a longer time horizon.

4. **TAU (0.001)**: This parameter is used for the soft update of the target network. It controls how much the target network's weights are updated towards the local network's weights. A small value like 0.001 ensures gradual updates, providing stability to the learning process.

5. **LR (0.0005)**: The learning rate defines how much the neural network weights are updated during training. A smaller learning rate ensures that learning is more stable but can slow down the learning process.

6. **UPDATE_EVERY (4)**: This parameter specifies how often the network is updated. In this case, the network updates every 4 time steps, balancing between learning frequently and computational efficiency.

7. **n_episodes (2000)**: The maximum number of training episodes. This sets the upper limit on how many times the agent will iterate through the environment during training.

8. **max_t (1000)**: The maximum number of timesteps per episode. It caps the length of each episode, ensuring episodes don’t run indefinitely.

9. **eps_start (1.0), eps_min (0.01), eps_decay (0.995)**: These parameters control the epsilon-greedy strategy for action selection. `eps_start` is the starting value of epsilon, `eps_min` is the minimum value, and `eps_decay` is the rate at which epsilon decays after each episode. This strategy balances exploration (trying new things) and exploitation (using known information).

## Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the course GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

### (Optional) Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the `p1_navigation/` folder in the course GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.
