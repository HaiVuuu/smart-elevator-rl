import os
import sys
import inspect

# Add the script's directory to the Python path to ensure modules can be found by Ray workers
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env import PettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from multi_agent_env import MARLElevatorEnv
from pettingzoo.utils.conversions import parallel_to_aec

def env_creator(config):
    parallel_env = MARLElevatorEnv(**config)
    aec_env = parallel_to_aec(parallel_env)
    return aec_env

register_env("marl_elevator", lambda config: PettingZooEnv(env_creator(config)))

# Create a dummy env to get the obs/action spaces for the shared policy
temp_env = MARLElevatorEnv()
ob_space = temp_env.observation_space(temp_env.possible_agents[0])
ac_space = temp_env.action_space(temp_env.possible_agents[0])
temp_env.close()


config = (
    PPOConfig()
    .environment(
        "marl_elevator",
        env_config={"render_mode": None, "sim_step_size": 1.0},
        disable_env_checking=True
    )
    .framework("torch")
    .env_runners(num_env_runners=3, rollout_fragment_length='auto')
    .training(
        gamma=0.995,
        lr=0.0005,
        clip_param=0.2,
        entropy_coeff=0.02,
        train_batch_size=4096,
        num_epochs=10
    )
    .multi_agent(
        policies={
            "shared_policy": PolicySpec(
                observation_space=ob_space,
                action_space=ac_space,
            )
        },
        policy_mapping_fn=(lambda agent_id, episode, **kwargs: "shared_policy"),
    )
    .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
)
config.sgd_minibatch_size = 512

algo = config.build()

checkpoint_dir = os.path.abspath("./models/marl_ppo/")
os.makedirs(checkpoint_dir, exist_ok=True)

# Training loop
print("Starting MARL Training with PPO")
print(f"Checkpoints will be saved in: {checkpoint_dir}")
print("Run 'tensorboard --logdir=~/ray_results' to monitor training.")

for i in range(200):
    result = algo.train()
    # Use .get() to avoid KeyError if the key is missing
    mean_reward = result.get("episode_reward_mean")
    timesteps_total = result.get("timesteps_total")

    if mean_reward is not None:
        print(f"Iteration: {i+1}, Mean Reward: {mean_reward:.2f}, Timesteps: {timesteps_total}")
    else:
        print(f"Iteration: {i+1}, Mean Reward: N/A, Timesteps: {timesteps_total}")

    if (i + 1) % 20 == 0:
        checkpoint_path = algo.save(checkpoint_dir)
        print(f"Checkpoint saved at: {checkpoint_path}")

final_checkpoint = algo.save(checkpoint_dir)
print("\nTraining Complete ")
print(f"Final model checkpoint saved to {final_checkpoint}")
