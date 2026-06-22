# Multi-Agent Reinforcement Learning (MARL) Elevator Training

This directory contains scripts for training and evaluating multi-agent reinforcement learning agents to control individual elevators in the smart elevator system.

## Overview

In the MARL approach:
- Each elevator is controlled by its own agent
- All agents share a common policy (single neural network)
- Agents are trained using Ray RLlib's PPO algorithm
- Communication and coordination emerge from the shared policy

## Requirements

Ensure you have installed the main project dependencies:
```bash
pip install -r ../requirements.txt
```

## Files

- **`multi_agent_env.py`** - PettingZoo environment for multi-agent training
- **`train_marl.py`** - Script to train multi-agent PPO policy
- **`evaluate_marl.py`** - Script to evaluate trained agents
- **`requirements.txt`** - MARL-specific dependencies (uses parent directory requirements)

## Usage

### Training

To train a multi-agent PPO policy:

```bash
python train_marl.py
```

**What happens:**
- Initializes a shared PPO policy for all elevators
- Trains for 200 iterations with Ray RLlib
- Saves checkpoints to `../models/marl_ppo/` every 20 iterations
- Prints mean reward and timesteps for each iteration
- Optionally monitor with TensorBoard (see below)

**Training Parameters (configurable in script):**
- `num_env_runners`: 3 (parallel environments)
- `lr`: 0.0005 (learning rate)
- `gamma`: 0.995 (discount factor)
- `train_batch_size`: 4096
- `num_epochs`: 10

### Evaluation

To evaluate the trained agent:

```bash
python evaluate_marl.py
```

**Configuration (in script):**
- `CHECKPOINT_DIR`: Path to saved checkpoints (default: `../models/marl_ppo/`)
- `NUM_EPISODES`: Number of episodes to run (default: 3)
- `RENDER_MODE`: Set to `"human"` to visualize, `None` for faster evaluation

**Output:**
- Number of steps per episode
- Number of delivered passengers
- Average passenger wait time
- Total cumulative reward

### Monitoring with TensorBoard

During training, Ray RLlib logs metrics to `~/ray_results/`. To visualize:

```bash
tensorboard --logdir=~/ray_results/
```

Then open http://localhost:6006 in your browser.

## Troubleshooting

### **Issue: "No checkpoint found"**
- Run `train_marl.py` first to generate checkpoints
- Verify `../models/marl_ppo/` directory exists and contains checkpoint files

### **Issue: Import errors for multi_agent_env**
- Ensure you're running scripts from the `marl/` directory
- Or add the parent directory to PYTHONPATH:
  ```bash
  export PYTHONPATH=$PYTHONPATH:..
  ```

### **Issue: GPU/CUDA errors**
- Set `RLLIB_NUM_GPUS=0` environment variable to use CPU only:
  ```bash
  export RLLIB_NUM_GPUS=0
  python train_marl.py
  ```

### **Issue: Out of memory**
- Reduce `train_batch_size` or `num_env_runners` in `train_marl.py`
- Use a machine with more RAM or reduce the number of training iterations

## Performance Tips

1. **Faster training:** Set `RENDER_MODE = None` in both scripts
2. **Better policy:** Increase `timesteps` in training loop
3. **Deterministic evaluation:** Set `explore=False` in `evaluate_marl.py` (already set)
4. **Multi-GPU:** Set `RLLIB_NUM_GPUS=2` (or your GPU count)

## Comparing with Single-Agent

Run the single-agent comparison script from the parent directory:
```bash
cd ..
python single_agent/compare_all_agents.py
```

This compares the single-agent PPO, single-agent DQN, and multi-agent policies.
