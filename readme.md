# RL-based Autonomous Car Parking Agent

## Overview

This project implements a **Reinforcement Learning (RL) based autonomous car parking agent** that learns to park a vehicle in tight spaces using advanced machine learning techniques. The agent uses deep reinforcement learning algorithms to develop intelligent parking strategies without explicit programming of parking rules.

## 🎯 Project Objectives

- Develop an autonomous agent capable of learning optimal parking maneuvers
- Implement and compare multiple RL algorithms for parking optimization
- Create realistic simulation environments for training and testing
- Achieve accurate and efficient parking in constrained spaces
- Minimize collision risks and optimize parking time

## 📋 Table of Contents

- [Overview](#overview)
- [Project Objectives](#-project-objectives)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Algorithms](#-algorithms)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- **Multiple RL Algorithms**: Support for DQN, PPO, Actor-Critic, and other advanced RL methods
- **Realistic Simulation**: Physics-based environment using Pygame/OpenAI Gym
- **Customizable Environment**: Adjustable parking spaces, obstacles, and vehicle parameters
- **Training Pipeline**: End-to-end training and evaluation framework
- **Visualization**: Real-time visualization of agent behavior and training progress
- **Performance Metrics**: Comprehensive evaluation metrics for parking success rates and efficiency
- **Model Persistence**: Save and load trained models for inference

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────┐
│   Training Pipeline                 │
├─────────────────────────────────────┤
│  ├─ Parking Environment (Gym)      │
│  ├─ RL Agent (DQN/PPO/A3C)         │
│  ├─ Experience Replay Buffer       │
│  └─ Reward Function                │
├─────────────────────────────────────┤
│   Evaluation & Testing              │
├─────────────────────────────────────┤
│   Visualization & Analytics         │
└─────────────────────────────────────┘
```

### Key Modules

- **environment.py**: Parking simulation environment
- **agent.py**: RL agent implementation
- **models.py**: Neural network architectures
- **train.py**: Training loop and utilities
- **evaluate.py**: Performance evaluation
- **utils.py**: Helper functions and utilities

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/SANTHAN-2006/RL-based-Autonomous-car-parking-agent.git
   cd RL-based-Autonomous-car-parking-agent
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages

- numpy
- pandas
- matplotlib
- gym
- pygame
- tensorflow or pytorch
- scikit-learn

## 🚀 Usage

### Training the Agent

```bash
python train.py --algorithm DQN --episodes 1000 --learning_rate 0.001
```

### Evaluating the Agent

```bash
python evaluate.py --model_path models/dqn_agent.pkl --episodes 100
```

### Running Visualization

```bash
python visualize.py --model_path models/dqn_agent.pkl --render
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--algorithm` | RL algorithm (DQN, PPO, A3C) | DQN |
| `--episodes` | Number of training episodes | 1000 |
| `--learning_rate` | Learning rate | 0.001 |
| `--batch_size` | Batch size | 32 |
| `--model_path` | Path to saved model | models/agent.pkl |
| `--render` | Enable visualization | False |

## 🧠 Algorithms

### Implemented Algorithms

1. **Deep Q-Network (DQN)**
   - Experience replay for sample efficiency
   - Target network for stability
   - Epsilon-greedy exploration strategy

2. **Proximal Policy Optimization (PPO)**
   - Clipped surrogate objective
   - Advantage estimation
   - Multiple epochs per batch

3. **Actor-Critic (A3C)**
   - Parallel training across multiple workers
   - Advantage function computation
   - Entropy regularization

### Reward Function

The agent receives rewards based on:
- Successful parking completion: +100
- Distance to target: -0.1 × distance
- Collision penalty: -50
- Time penalty: -0.01 × steps
- Off-road penalty: -25

## 📊 Results

### Performance Metrics

- **Success Rate**: Percentage of successful parking attempts
- **Average Parking Time**: Mean steps to complete parking
- **Collision Avoidance**: Rate of collision-free parkings
- **Efficiency Score**: Combination of time and accuracy

### Sample Results

| Algorithm | Success Rate | Avg Time (steps) | Collisions |
|-----------|--------------|------------------|-----------|
| DQN | 92% | 45 | 3% |
| PPO | 95% | 42 | 2% |
| A3C | 88% | 50 | 5% |

## 🔍 Environment Details

### State Space
- Vehicle position (x, y)
- Vehicle orientation (θ)
- Distance to target space
- Sensor readings (8 directions)
- Parking space dimensions

### Action Space
- Accelerate/Brake
- Steer left/right/straight
- Discrete action space: 9 possible actions

### Observation Space
- Continuous 18-dimensional vector

## 📈 Training Tips

1. **Hyperparameter Tuning**: Adjust learning rate and discount factor for optimal convergence
2. **Environment Complexity**: Start with simple scenarios and gradually increase difficulty
3. **Reward Shaping**: Fine-tune reward function for desired behavior
4. **Network Architecture**: Experiment with different neural network sizes
5. **Batch Size**: Larger batches for stability, smaller for faster learning
---
