from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
import numpy as np


class BuildingMultiAgentEnv(ParallelEnv):

    def __init__(self, building):

        self.building = building

        self.num_elevators = len(building.elevators)
        self.num_floors = building.num_floors

        self.agents = [f"elevator_{i}" for i in range(self.num_elevators)]
        self.observation_spaces = {

            agent: spaces.Box(low=0, high=self.num_floors, shape=(self.num_floors+4,), dtype=np.float32)
            for agent in self.agents

        }

        self.action_spaces = {

            agent: spaces.Discrete(3)  # stay, up, down
            for agent in self.agents

        }


    def reset(self, seed=None, options=None):
        """
        Docstring here
        """
        self.building.reset()

        return {agent: self._get_obs(agent) for agent in self.agents}


    def elevator_state_to_numpy(self,elevator):
        floor = elevator['floor'] / (self.num_floors - 1)

        dir_map = {-1: 0, 0: 1, 1: 2}
        direction = np.zeros(3, dtype=np.float32)
        direction[dir_map[elevator['direction']]] = 1

        passenger_counts = np.zeros(self.num_floors, dtype=np.float32)
        for dest in elevator['passengers_dest']:
            passenger_counts[dest] += 1

        return np.concatenate(([floor], direction, passenger_counts))


    def _get_obs(self, agent):
        """
        Docstring here
        """
        idx = int(agent.split("_")[1])
        elevator_state = self.building.get_elevator_state(idx)
        elevator_state = self.elevator_state_to_numpy(elevator_state)

        return elevator_state


    def step(self, actions):
        """
        Docstring here
        """
        state, rewards, done, infos = self.building.step(actions)

        obs = {agent: self._get_obs(agent) for agent in self.agents}
        rewards = {agent: rewards[i] for i, agent in enumerate(self.agents)}
        dones = {agent: done for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return obs, rewards, dones, infos
