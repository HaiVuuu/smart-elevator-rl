# Contributing to Smart Elevator

Thank you for your interest in contributing! Here's how to get started.

## Getting Started

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/yourusername/Smart-Elevator.git
   cd Smart-Elevator
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Areas

### 1. Core Simulation (`building.py`, `elevator.py`, `person.py`)
- Optimize physics and movement logic
- Add new building features (doors, traffic patterns)
- Improve performance for large buildings

### 2. Single-Agent RL (`single_agent/`)
- Experiment with new algorithms (PPO, DQN variants)
- Tune hyperparameters for better convergence
- Add new baseline algorithms
- Improve visualization and logging

### 3. Multi-Agent RL (`marl/`)
- Fix any convergence issues
- Experiment with different policy architectures
- Add communication mechanisms between agents
- Improve environment rewards

### 4. Visualization & Analysis
- Enhance `view.py` rendering
- Add new plots in analysis scripts
- Create performance dashboards

## Testing

Before submitting a PR:

1. **Test single-agent training:**
   ```bash
   python single_agent/train.py --algo ppo --lr 0.0001 --timesteps 10000
   ```

2. **Test evaluation:**
   ```bash
   python single_agent/compare_all_agents.py
   ```

3. **Test multi-agent training:**
   ```bash
   cd marl && python train_marl.py
   ```

4. **Check for import errors:**
   ```bash
   python -c "from building import Building; from environment import *"
   ```

## Code Style

- Follow PEP 8 conventions
- Use descriptive variable names
- Add docstrings to complex functions
- Keep functions focused and modular

## Submitting Changes

1. **Commit with clear messages:**
   ```bash
   git commit -m "Add: meaningful description of changes"
   ```

2. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** with:
   - Clear title describing the change
   - Description of what and why
   - Any related issues (e.g., "Fixes #123")
   - Testing performed

## Reporting Issues

When reporting bugs, please include:
- Python version and OS
- Steps to reproduce
- Expected vs. actual behavior
- Relevant error messages/logs

## Questions?

- Check existing issues and discussions
- Review the READMEs in respective directories
- Open a new issue to discuss ideas

Thanks for contributing!
