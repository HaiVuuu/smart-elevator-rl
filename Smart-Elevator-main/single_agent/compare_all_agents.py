import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from single_agent_env import SingleAgentElevatorEnv
from flatten_action_wrapper import FlattenActionWrapper
from normal_algorithm import NormalAlgorithm
from utils import evaluate_model

# --- Configuration ---
NUM_EPISODES = 1
MAX_STEPS = 50000
SIM_STEP_SIZE = 1.0
RENDER_MODE = None # Set to None for faster evaluation

AGENTS_TO_COMPARE = [
    {
        "name": "PPO",
        "model_loader": PPO,
        "model_path": "models/Single_agents/LR_00005/ppo_elevator.zip",
        "env_wrapper": None,
        "eval_params": {"deterministic": False, "delay": 0}
    },
    {
        "name": "DQN",
        "model_loader": DQN,
        "model_path": "models/Single_agents/LR_00005/dqn_elevator.zip",
        "env_wrapper": FlattenActionWrapper,
        "eval_params": {"deterministic": True, "delay": 0}
    },
    {
        "name": "Rule-Based",
        "model_loader": NormalAlgorithm,
        "model_path": None, # Rule-based doesn't load from a file
        "env_wrapper": None,
        "eval_params": {"deterministic": True, "delay": 0}
    }
]

# --- Evaluation Loop ---
evaluation_results = {agent["name"]: [] for agent in AGENTS_TO_COMPARE}

for ep in range(NUM_EPISODES):
    print(f"\n--- Starting Episode {ep + 1}/{NUM_EPISODES} ---")

    for agent_config in AGENTS_TO_COMPARE:
        agent_name = agent_config["name"]
        print(f"  Evaluating {agent_name}...")

        # Create environment for the agent
        env = SingleAgentElevatorEnv(render_mode=RENDER_MODE, sim_step_size=SIM_STEP_SIZE)
        if agent_config["env_wrapper"]:
            env = agent_config["env_wrapper"](env)

        # Load or initialize model
        if agent_config["model_path"]:
            model = agent_config["model_loader"].load(agent_config["model_path"], env=env)
        else:
            model = agent_config["model_loader"]()

        # Evaluate and store results
        info = evaluate_model(model, env, MAX_STEPS, **agent_config["eval_params"])
        evaluation_results[agent_name].append(info.get('avg_wait', -1))
        print(f"  Ep {ep + 1} {agent_name}: Avg Wait {info.get('avg_wait', -1):.2f}, Delivered {info.get('delivered', 0)}")
        env.close()

# --- Plotting Results ---
plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(12, 7))

for name, avg_waits in evaluation_results.items():
    if any(wait > 0 for wait in avg_waits): # Plot only if there is valid data
        ax.plot(range(1, NUM_EPISODES + 1), avg_waits, label=name, marker='o', linestyle='-')

ax.set_xlabel("Episode")
ax.set_ylabel("Average Wait Time (s)")
ax.set_title("Performance Evaluate with 0.0001 Learning Rate Agent: PPO vs. DQN vs. Rule-Based")
ax.legend()
fig.patch.set_facecolor('#f0f0f0')
plt.tight_layout()
plt.show()
