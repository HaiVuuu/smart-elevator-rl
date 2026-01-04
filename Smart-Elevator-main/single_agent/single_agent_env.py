import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from base_env import BaseElevatorEnv

class SingleAgentElevatorEnv(gym.Env, BaseElevatorEnv):
    def __init__(self, render_mode=None, sim_step_size=1.0):
        # Initialize the base environment
        BaseElevatorEnv.__init__(self, render_mode=render_mode, sim_step_size=sim_step_size)
        
        # Define Gym-specific action and observation spaces
        self.action_space = spaces.MultiDiscrete([3] * self.num_elevators)
        
        elevator_state_size = self.num_elevators * (1 + 3 + self.num_floors)
        waiting_size = self.num_floors * 2 
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(elevator_state_size + waiting_size,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset_building()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        actions = action.tolist()
        _, update_infos, done, _ = self.building.step(actions)
        
        total_reward = 0
        
        for i, info in enumerate(update_infos):
            reward = 0
            if info['is_idle'] and (any(self.building.waiting_people) or info['num_passengers'] > 0):
                reward -= 1.0
            reward += 50.0 * info['passengers_dropped_off']
            reward += 10.0 * info['passengers_picked_up']
            reward -= 0.05 * info['num_passengers']
            total_reward += reward

        num_waiting_on_floors = sum(len(q) for q in self.building.waiting_people)
        total_reward -= 0.05 * num_waiting_on_floors
        
        obs = self._get_obs()
        
        delivered_count = self.building.delivered_people_count
        if delivered_count > 0:
            avg_wait = self.building.total_wait_time / delivered_count
            avg_pickup_wait_time = self.building.get_average_pickup_wait_time()
            avg_travel_time = self.building.get_average_travel_time()
        else:
            avg_wait = float('inf')
            avg_pickup_wait_time = float('inf')
            avg_travel_time = float('inf')

        info = {
            "delivered": delivered_count, 
            "avg_wait": avg_wait,
            "avg_pickup_wait_time": avg_pickup_wait_time,
            "avg_travel_time": avg_travel_time
        }
        return obs, total_reward, done, False, info

    def _get_obs(self):
        elevator_states = [self._get_local_obs_part(i) for i in range(self.num_elevators)]
        global_state = self._get_global_obs_part()
        
        # For single-agent, flatten everything into one vector
        obs = np.concatenate(elevator_states + [global_state])
        return obs
