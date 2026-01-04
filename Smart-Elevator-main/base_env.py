import pygame
import json
import numpy as np

from building import Building
from view import BuildingView
from constants import ScreenConfig

with open('params.json', 'r') as f:
    params = json.load(f)

class BaseElevatorEnv:
    """
    A base environment that contains the common logic for both the single-agent (Gym)
    and multi-agent (PettingZoo) elevator environments.
    """
    def __init__(self, render_mode=None, sim_step_size=1.0):
        pygame.init()
        self.render_mode = render_mode
        self.screen = pygame.display.set_mode((ScreenConfig.WIDTH, ScreenConfig.HEIGHT)) if render_mode == "human" else pygame.Surface((ScreenConfig.WIDTH, ScreenConfig.HEIGHT))
        self.font = pygame.font.Font(None, 24)
        
        floor_height = self.screen.get_height() // (params['num_floors'] + 1)
        
        self.building = Building(floor_height=floor_height, **params, sim_step_size=sim_step_size)
        
        self.building_view = None
        if self.render_mode == "human":
            self.building_view = BuildingView(self.building, self.screen, self.font)

        self.num_floors = self.building.num_floors
        self.num_elevators = self.building.num_elevators

    def _get_global_obs_part(self):
        """Calculates the global part of the observation (waiting people)."""
        up_requests = np.zeros(self.num_floors)
        down_requests = np.zeros(self.num_floors)
        for f in range(self.num_floors):
            up_count = sum(1 for p in self.building.waiting_people[f] if p.destination_floor > f)
            down_count = sum(1 for p in self.building.waiting_people[f] if p.destination_floor < f)
            up_requests[f] = up_count
            down_requests[f] = down_count
        
        normalization_factor = 10.0
        up_requests /= normalization_factor
        down_requests /= normalization_factor
        return np.concatenate([up_requests, down_requests])

    def _get_local_obs_part(self, elevator_idx):
        """Calculates the local part of the observation for a single elevator."""
        elev = self.building.get_elevator_state(elevator_idx)
        floor_norm = elev['floor'] / (self.num_floors - 1)
        
        dir_onehot = np.zeros(3)
        dir_map = {-1: 0, 0: 1, 1: 2}
        dir_onehot[dir_map[elev['direction']]] = 1
        
        pass_counts = np.zeros(self.num_floors)
        for dest in elev['passengers_dest']:
            pass_counts[dest] += 1
        pass_counts /= self.building.elevator_capacity # Normalize

        return np.concatenate(([floor_norm], dir_onehot, pass_counts))

    def reset_building(self):
        """Resets the underlying building simulation."""
        self.building.reset()

    def render(self):
        """Renders the environment."""
        if self.render_mode == "human":
            self.building_view.draw()

    def close(self):
        """Closes the environment and quits Pygame."""
        pygame.quit()
