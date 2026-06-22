# Setup Guide for Smart Elevator

Detailed installation and setup instructions for all platforms.

## Prerequisites

- **Python 3.11+** (check with `python --version`)
- **Git** (for cloning)
- **8GB RAM minimum** (16GB+ recommended for multi-agent training)
- **2GB disk space** (for dependencies and models)

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/Smart-Elevator.git
cd Smart-Elevator
```

### 2. Create Virtual Environment

#### On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### On Windows (Command Prompt):
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

#### On Windows (PowerShell):
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** This includes:
- `ray[rllib]` - Multi-agent RL framework
- `gymnasium` - RL environment standard
- `stable-baselines3` - Single-agent algorithms
- `torch` - Neural network backend
- `pygame` - Visualization
- `matplotlib` - Plotting

### 4. Verify Installation

```bash
python -c "import gymnasium; import torch; import ray; print('✓ All dependencies installed')"
```

## Troubleshooting Installation

### Issue: `ModuleNotFoundError: No module named 'ray'`
**Solution:** Reinstall requirements
```bash
pip install -r requirements.txt --force-reinstall
```

### Issue: `RuntimeError: No CUDA runtime found` (GPU error on CPU-only machine)
**Solution:** PyTorch will fall back to CPU automatically. No action needed.

### Issue: `pygame.error: No available video device`
**Solution:** This is normal on headless systems. Set `render_mode="None"` in scripts.

### Issue: `ImportError: cannot import name 'parallel_to_aec'`
**Solution:** Update pettingzoo
```bash
pip install pettingzoo --upgrade
```

### Issue: Out of memory during training
**Solutions:**
- Reduce `train_batch_size` in `marl/train_marl.py`
- Reduce `num_env_runners`
- Reduce episode length
- Use CPU instead of GPU (`export RLLIB_NUM_GPUS=0`)

## Quick Start

### Training a Single-Agent

```bash
python single_agent/train.py --algo ppo --lr 0.0001 --timesteps 100000
```

Trained model saved to: `models/Single_agents/LR_0.0001/`

### Training Multi-Agent

```bash
cd marl
python train_marl.py
```

Checkpoints saved to: `../models/marl_ppo/`

### Evaluating Models

```bash
# Single-agent comparison
python single_agent/compare_all_agents.py

# Multi-agent evaluation
cd marl
python evaluate_marl.py
```

## Environment Variables

Useful environment variables for customization:

```bash
# Use CPU only (no GPU)
export RLLIB_NUM_GPUS=0

# Use 2 GPUs for training
export RLLIB_NUM_GPUS=2

# Disable Pygame display (headless)
export SDL_VIDEODRIVER=dummy

# Verbosity level (0=quiet, 1=normal, 2=verbose)
export PYTHONVERBOSITY=2
```

## Running Tests

Quick validation:

```bash
# Test imports
python -c "from building import Building; from environment import SingleAgentElevatorEnv; print('✓ Core modules OK')"

# Test MARL imports
python -c "from marl.multi_agent_env import MARLElevatorEnv; print('✓ MARL module OK')"

# Test single-agent training (short)
python single_agent/train.py --algo ppo --lr 0.01 --timesteps 1000
```

## GPU Setup (Optional)

For faster training with NVIDIA GPU:

1. Install NVIDIA CUDA Toolkit (11.8+)
2. Install cuDNN
3. PyTorch will detect GPU automatically

Verify GPU:
```bash
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

## Next Steps

- Read the main [README.md](README.md)
- Explore single-agent training in [single_agent/](single_agent/)
- Try multi-agent training in [marl/](marl/)
- Review [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting-installation) section above
2. Review script-specific READMEs in subdirectories
3. Search existing GitHub issues
4. Create a new issue with your error details

Happy training! 🚀
