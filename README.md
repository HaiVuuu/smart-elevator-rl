# 🚀 Smart Elevator: Reinforcement Learning Simulator

This project provides a simulation environment for training and evaluating reinforcement learning agents to optimize elevator dispatch strategies. The goal is to minimize passenger wait times and improve building efficiency, especially during peak hours.

The environment supports both single-agent (controlling all elevators) and multi-agent (one agent per elevator) reinforcement learning approaches.

---

## 🏗️ Project Structure

```
.
├── Smart-Elevator-main/
│   ├── single_agent/     # Scripts for single-agent RL (PPO, DQN)
│   ├── marl/             # Scripts for multi-agent RL (PPO with shared policy)
│   ├── models/           # Stores trained model checkpoints
│   ├── environment.py    # Core simulation environment
│   ├── requirements.txt  # Project dependencies
│   └── ...
└── ...
```

---

## ⚙️ Installation

### 1. Prerequisites
- Python 3.11 or later

### 2. Setup
1.  **Clone the repository:**
    ```bash
    git clone 
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # On Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # On macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚡ How to Use

### Single-Agent RL

The single-agent approach uses one central agent to control all elevators in the system.

#### 1. Training a Single Agent
Use the `single_agent/train.py` script to train a PPO or DQN agent.

**Usage:**
```bash
python single_agent/train.py --algo <ppo|dqn> --lr <learning_rate> --timesteps <total_timesteps>
```

**Example:** Train a PPO agent with a learning rate of `0.0001` for `1,000,000` timesteps.
```bash
python single_agent/train.py --algo ppo --lr 0.0001 --timesteps 1000000
```
- Trained models and checkpoints are saved in `models/Single_agents/LR_<learning_rate_value>/`.
- Training progress can be monitored using TensorBoard (see below).

#### 2. Evaluating Single Agents
The `single_agent/compare_all_agents.py` script runs a comparison between a trained PPO agent, a DQN agent, and the baseline rule-based algorithm.

**Usage:**
```bash
python single_agent/compare_all_agents.py
```
- **Note:** This script uses hardcoded paths to pre-trained models stored in `models/Single_agents/`. To evaluate your own models, you must edit the `AGENTS_TO_COMPARE` list within the script to point to your `.zip` model files.
- The script will display a plot comparing the average passenger wait times for each agent.

---

### Multi-Agent RL (MARL)

The multi-agent approach assigns an independent agent to control each elevator, sharing a common policy.

#### 1. Training MARL Agents
Run the `marl/train_marl.py` script to start training. The hyperparameters are configured directly within the script.

**Usage:**
```bash
python marl/train_marl.py
```
- Checkpoints are saved periodically to `models/marl_ppo/`.
- Training progress can be monitored using TensorBoard (see Ray/RLlib instructions below).

#### 2. Evaluating MARL Agents
The evaluation script automatically loads the latest trained checkpoint and runs the simulation. You can enable visualization by setting `RENDER_MODE = "human"` inside the script.

**Usage:**
```bash
python marl/evaluate_marl.py
```
- The script prints key performance metrics, such as the number of delivered people and the average wait time for each evaluation episode.

---

## 📊 Monitoring Training with TensorBoard

You can visualize training metrics (like rewards and episode lengths) using TensorBoard.

-   **For Single-Agent Training:**
    ```bash
    tensorboard --logdir=logs
    ```

-   **For Multi-Agent (RLlib) Training:**
    RLlib creates its own log directory.
    ```bash
    tensorboard --logdir=~/ray_results
    ```

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
