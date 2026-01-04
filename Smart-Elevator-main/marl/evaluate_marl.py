import os
import sys
import inspect
import time
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from pettingzoo.utils.conversions import parallel_to_aec
from multi_agent_env import MARLElevatorEnv

# --- Configuration ---
CHECKPOINT_DIR = "./models/Multi_agents/marl_ppo/"
NUM_EPISODES = 3
RENDER_MODE = "human" # "human" to watch, None for faster evaluation
DELAY = 0.05 # Delay between steps in human render mode

# --- Environment Setup ---
def env_creator(config):
    parallel_env = MARLElevatorEnv(**config)
    aec_env = parallel_to_aec(parallel_env)
    return aec_env

register_env("marl_elevator", lambda config: PettingZooEnv(env_creator(config)))

# --- Find latest checkpoint ---
def find_latest_checkpoint(directory):
    """Finds the latest RLLib checkpoint in a directory."""
    latest_checkpoint = None
    latest_mtime = 0
    if not os.path.exists(directory):
        return None
        
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".rllib_checkpoint"):
                checkpoint_path = os.path.join(root, f)
                mtime = os.path.getmtime(checkpoint_path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    # RLLib expects the directory containing the checkpoint file
                    latest_checkpoint = root
    return latest_checkpoint

# --- Main Evaluation Logic ---
if __name__ == "__main__":
    latest_checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)

    if not latest_checkpoint_path:
        print(f"Error: No checkpoint found in directory {CHECKPOINT_DIR}")
        print("Please make sure you have trained a multi-agent model using 'train_marl.py' first.")
        sys.exit(1)

    print(f"Loading model from checkpoint: {latest_checkpoint_path}")

    # Load the trained algorithm
    algo = Algorithm.from_checkpoint(latest_checkpoint_path)

    # Create the environment for evaluation
    env = MARLElevatorEnv(render_mode=RENDER_MODE)

    print("\n--- Starting Evaluation ---")
    for ep in range(NUM_EPISODES):
        print(f"\n--- Episode {ep + 1}/{NUM_EPISODES} ---")
        
        terminations = {agent: False for agent in env.possible_agents}
        truncations = {agent: False for agent in env.possible_agents}
        obs, info = env.reset()
        
        total_reward = {agent: 0 for agent in env.possible_agents}
        steps = 0

        while not (any(terminations.values()) or any(truncations.values())):
            actions = {}
            for agent_id in env.agents:
                action = algo.compute_single_action(
                    observation=obs[agent_id],
                    policy_id="shared_policy",
                    explore=False # Set to False for deterministic evaluation
                )
                actions[agent_id] = action

            obs, reward, terminations, truncations, info = env.step(actions)
            
            for agent_id, r in reward.items():
                total_reward[agent_id] += r

            if RENDER_MODE == "human":
                env.render()
                time.sleep(DELAY)
            
            steps += 1

        building_stats = info.get("building_stats", {})
        print(f"Episode finished after {steps} steps.")
        print(f"  - Delivered People: {building_stats.get('delivered_people_count', 'N/A')}")
        print(f"  - Average Wait Time: {building_stats.get('average_wait_time', 'N/A'):.2f}s")
        print(f"  - Total Rewards: {sum(total_reward.values()):.2f}")

    env.close()
    print("\n--- Evaluation Complete ---")
