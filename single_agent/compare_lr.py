import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from single_agent_env import SingleAgentElevatorEnv
from flatten_action_wrapper import FlattenActionWrapper
from utils import evaluate_model
from pathlib import Path

def evaluate(env, model, num_episodes, max_steps, deterministic=False, delay=0, render=True):
    """Evaluates a model for a given number of episodes and returns the average wait times."""
    wait_times = []
    for ep in range(num_episodes):
        print(f"\n Starting Episode {ep + 1}/{num_episodes} ")
        info = evaluate_model(model, env, max_steps, deterministic, delay, render)
        avg_wait = info.get('avg_wait', -1)
        print(f"Ep {ep + 1} Avg Wait {avg_wait:.2f}, Delivered {info.get('delivered', 0)}")
        wait_times.append(avg_wait)
    return wait_times


def compare_learning_rate(models, env, num_episodes, max_steps):
    """
    Compares different models based on their learning rates over several episodes.
    """
    results = {}
    for name, model in models.items():
        print(f"\n Evaluating model: {name} ")
        wait_times = evaluate(env, model, num_episodes, max_steps, render=False)
        results[name] = wait_times
    return results

def plot_results(results, agent_name, num_episodes):
    """Plots the evaluation results."""
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))
 
    for name, avg_waits in results.items():
        if any(wait > 0 for wait in avg_waits):  # Plot only if there is valid data
            ax.plot(range(1, num_episodes + 1), avg_waits, label=name, marker='o', linestyle='-')
 
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Wait Time (s)")
    ax.set_title(f"Learning Rate Effect on Agent Performance with {agent_name} Agent")
    ax.legend()
    fig.patch.set_facecolor('#f0f0f0')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    num_episodes = 5
    max_steps = 50000
    learning_rates = ["LR_00001","LR_00005","LR_0001"]

    ppo_env = SingleAgentElevatorEnv(render_mode="human", sim_step_size=1.0)
    dqn_env = FlattenActionWrapper(SingleAgentElevatorEnv(render_mode="human", sim_step_size=1.0))

    models_path = Path(__file__).parent.parent / "models" / "Single_agents"

    ppo_models = {lr: PPO.load(models_path / lr / "ppo_elevator", env=ppo_env) for lr in learning_rates}
    ppo_results = compare_learning_rate(ppo_models,ppo_env,num_episodes,max_steps)
    plot_results(ppo_results, "PPO", num_episodes)


    dqn_models = {lr: DQN.load(models_path / lr / "dqn_elevator", env=dqn_env) for lr in learning_rates}
    dqn_results = compare_learning_rate(dqn_models,dqn_env,num_episodes,max_steps)
    plot_results(dqn_results, "DQN", num_episodes)
    # plt.style.use('seaborn-v0_8-darkgrid')
    # fig, ax = plt.subplots(figsize=(12, 7))

    # for name, avg_waits in evaluation_results.items():
    #     if any(wait > 0 for wait in avg_waits): # Plot only if there is valid data
    #         ax.plot(range(1, num_episodes + 1), avg_waits, label=name, marker='o', linestyle='-')

    # ax.set_xlabel("Episode")
    # ax.set_ylabel("Average Wait Time (s)")
    # ax.set_title("Learning Rate Affectd on Agent Performance with DQN Agent")
    # ax.legend()
    # fig.patch.set_facecolor('#f0f0f0')
    # plt.tight_layout()
    # plt.show()
