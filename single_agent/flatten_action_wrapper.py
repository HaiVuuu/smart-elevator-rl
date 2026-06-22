import gymnasium as gym
import numpy as np

# Wrapper to convert MultiDiscrete action space to Discrete for DQN
class FlattenActionWrapper(gym.ActionWrapper):

    def __init__(self, env):

        super(FlattenActionWrapper, self).__init__(env)
        # Calculate the total number of discrete actions
        self.num_actions = self.action_space.nvec.prod()
        self.action_space = gym.spaces.Discrete(self.num_actions)
        self.action_shape = self.action_space.shape

    def action(self, action):
        # Convert the single discrete action back to the MultiDiscrete format
        return np.array(np.unravel_index(action, self.env.action_space.nvec))

    def reverse_action(self, action):
        # This is not needed for training, but good practice to have
        return np.ravel_multi_index(action, self.env.action_space.nvec)
