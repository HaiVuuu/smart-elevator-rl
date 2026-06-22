import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import lru_cache
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pettingzoo import ParallelEnv

from base_env import BaseElevatorEnv

class MARLElevatorEnv(ParallelEnv, BaseElevatorEnv):

    metadata = {"render_modes": ["human"], "name": "elevator_marl_v0"}

    def __init__(self, render_mode=None, sim_step_size=1.0):
        # Initialize the base environment
        BaseElevatorEnv.__init__(self, render_mode=render_mode, sim_step_size=sim_step_size)

        # PettingZoo API attributes
        self.possible_agents = [f"elevator_{i}" for i in range(self.num_elevators)]
        self.agents = []

        self.observation_spaces = {agent: self.observation_space(agent) for agent in self.possible_agents}
        self.action_spaces = {agent: self.action_space(agent) for agent in self.possible_agents}

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        elevator_state_size = 1 + 3 + self.num_floors
        waiting_size = self.num_floors * 2
        return spaces.Box(low=-np.inf, high=np.inf, shape=(elevator_state_size + waiting_size,), dtype=np.float32)

    @lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset_building()
        self.agents = self.possible_agents[:]
        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        action_list = [actions[agent] for agent in self.possible_agents]
        _, update_infos, done, _ = self.building.step(action_list)

        rewards = {}
        num_waiting_on_floors = sum(len(q) for q in self.building.waiting_people)
        system_penalty = (0.05 * num_waiting_on_floors) / self.num_elevators

        for i, agent in enumerate(self.possible_agents):
            info = update_infos[i]
            agent_reward = 0
            if info['is_idle'] and (any(self.building.waiting_people) or info['num_passengers'] > 0):
                agent_reward -= 1.0
            agent_reward += 50.0 * info['passengers_dropped_off']
            agent_reward += 10.0 * info['passengers_picked_up']
            agent_reward -= 0.05 * info['num_passengers']
            agent_reward -= system_penalty
            rewards[agent] = agent_reward

        terminations = {agent: done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        
        observations = self._get_obs()
        
        infos = {agent: {} for agent in self.agents}
        if done:
            delivered_count = self.building.delivered_people_count
            avg_wait = self.building.total_wait_time / delivered_count if delivered_count > 0 else float('inf')
            for agent in self.agents:
                infos[agent]['avg_wait'] = avg_wait
                infos[agent]['delivered'] = delivered_count

        if any(terminations.values()) or any(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self):
        global_state = self._get_global_obs_part()
        observations = {}
        for i, agent_id in enumerate(self.possible_agents):
            local_state = self._get_local_obs_part(i)
            observations[agent_id] = np.concatenate([local_state, global_state]).astype(np.float32)
        return observations