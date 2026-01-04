import argparse
import json
import os
from pathlib import Path

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from single_agent_env import SingleAgentElevatorEnv
from flatten_action_wrapper import FlattenActionWrapper

def train_agent(algo, lr, total_timesteps):
    """
    Trains a single-agent reinforcement learning model.

    Args:
        algo (str): The algorithm to use ('ppo' or 'dqn').
        lr (float): The learning rate.
        total_timesteps (int): The total number of training timesteps.
    """
    
    algo_map = {
        "ppo": (PPO, None),
        "dqn": (DQN, FlattenActionWrapper)
    }

    if algo.lower() not in algo_map:
        raise ValueError(f"Algorithm '{algo}' not supported. Choose from {list(algo_map.keys())}")

    model_class, wrapper = algo_map[algo.lower()]
    
    # --- Environment Setup ---
    env_kwargs = {"render_mode": None, "sim_step_size": 1.0}
    env = make_vec_env(SingleAgentElevatorEnv, n_envs=4, env_kwargs=env_kwargs, wrapper_class=wrapper)

    # --- Paths and Callbacks ---
    model_dir = Path(f"models/Single_agents/LR_{str(lr).split('.')[-1]}")
    log_dir = Path("logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
      save_freq=100000,
      save_path=str(model_dir),
      name_prefix=f'{algo.lower()}_elevator_checkpoint'
    )
    
    # --- Model Hyperparameters ---
    # Using default hyperparameters from the original scripts for now
    if algo.lower() == "ppo":
        model_params = {
            "learning_rate": lr,
            "n_steps": 2048,
            "batch_size": 512,
            "n_epochs": 10,
            "gamma": 0.995,
            "ent_coef": 0.02,
            "clip_range": 0.2,
        }
    else: # dqn
        model_params = {
            "learning_rate": lr,
            "buffer_size": 1_000_000,
            "learning_starts": 50000,
            "batch_size": 256,
            "gamma": 0.995,
            "train_freq": (4, "step"),
            "gradient_steps": 1,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.05,
            "policy_kwargs": dict(net_arch=[256, 256]),
        }

    # --- Model Initialization and Training ---
    model = model_class(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(log_dir),
        **model_params
    )

    print(f"\n--- Starting Training for {algo.upper()} with LR={lr} ---")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=f"{algo.upper()}_{lr}"
    )

    # --- Save Final Model ---
    final_model_path = model_dir / f"{algo.lower()}_elevator.zip"
    model.save(final_model_path)
    
    print("\n--- Training Complete ---")
    print(f"Final model saved to: {final_model_path}")
    print(f"Checkpoints saved in: {model_dir}")
    print("To monitor training, run the following command in your terminal:")
    print(f"tensorboard --logdir={log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent for the Smart Elevator.")
    parser.add_argument("--algo", type=str, required=True, help="Algorithm to use ('ppo' or 'dqn')")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for the optimizer.")
    parser.add_argument("--timesteps", type=int, required=True, help="Total number of training timesteps.")
    
    args = parser.parse_args()
    
    train_agent(args.algo, args.lr, args.timesteps)
