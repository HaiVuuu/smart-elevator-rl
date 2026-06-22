from constants import ElevatorConfig
from elevator import Elevator
from person import Person

import random 


class Building:

    def __init__(
            self,
            floor_height: int,
            num_floors: int,
            num_elevators: int,
            elevator_capacity: int,
            spawn_weight: list[float],
            need_to_carry: int = 50,
            day_delay: float = 10,
            elevator_delay: float = 0.1,  # Giảm xuống để di chuyển nhanh hơn trong sim
            sim_step_size: float = 0.01  # Thêm arg để tune sim speed (lớn = spawn/di chuyển nhanh hơn)
    ):

        self.num_floors = num_floors
        self.floor_height = floor_height

        self.num_elevators = num_elevators
        self.elevator_capacity = elevator_capacity
        self.elevator_delay = elevator_delay

        self.need_to_carry = need_to_carry

        self.spawn_weight = spawn_weight
        self.day_convert = {
            1: 'Monday',
            2: 'Tuesday',
            3: 'Wednesday',
            4: 'Thursday',
            5: 'Friday',
            6: 'Saturday',
            0: 'Sunday'
        }
        self.current_day = 1
        self.day_delay = day_delay

        self.sim_time = 0.0  # Simulated time mới (di chuyển lên trước elevators)
        self.sim_step_size = sim_step_size  
        self.last_spawn_time = self.sim_time
        self.day_timer = self.sim_time

        self.elevators = [Elevator(self, ElevatorConfig.WIDTH, ElevatorConfig.HEIGHT, ElevatorConfig.DOOR_WIDTH, elevator_capacity, elevator_delay) for _ in range(num_elevators)]

        self.waiting_people = [[] for _ in range(num_floors)]

        self.delivered_people_count = 0
        self.total_wait_time = 0
        self.total_travel_time = 0
        self.total_pickup_wait_time = 0

    def get_average_pickup_wait_time(self):
        if self.delivered_people_count == 0:
            return 0
        return self.total_pickup_wait_time / self.delivered_people_count

    def get_average_travel_time(self):
        if self.delivered_people_count == 0:
            return 0
        return self.total_travel_time / self.delivered_people_count

    def reset(self):
        """
        Resets the environment for a new episode.
        """
        self.sim_time = 0.0  # Reset sim_time (di chuyển lên trước elevators)
        self.last_spawn_time = self.sim_time
        self.day_timer = self.sim_time
        self.current_day = 1

        self.elevators = [
            Elevator(
                self,
                ElevatorConfig.WIDTH,
                ElevatorConfig.HEIGHT,
                ElevatorConfig.DOOR_WIDTH,
                self.elevator_capacity,
                self.elevator_delay 
            )
            for _ in range(self.num_elevators)
        ]
        self.waiting_people = [[] for _ in range(self.num_floors)]
        self.delivered_people_count = 0
        self.total_wait_time = 0
        self.total_travel_time = 0

        return self.get_state()

    def get_elevator_state(self, idx) -> dict:
        """
        Docstring here
        """
        elevator = self.elevators[idx]
        state = {
            'floor': elevator.floor,
            'direction': elevator.direction,
            'passengers_dest': [p.destination_floor for p in elevator.passengers]
        }

        return state

    def get_state(self):
        """
        Returns the current state for the RL agent.
        """
        state = {}
        for i, elevator in enumerate(self.elevators):

            state[f'elevator_{i}_state'] = {
                'floor': elevator.floor,
                'direction': elevator.direction,
                'passengers_dest': [p.destination_floor for p in elevator.passengers],
            }
        state['waiting_people'] = [[p.destination_floor for p in floor_list] for floor_list in self.waiting_people]

        return state

    def spawn_people(self, spawn_frequency: float) -> None:
        """
        Docstring here
        """
        if self.sim_time - self.last_spawn_time > spawn_frequency:
            start_floor = random.randint(0, self.num_floors - 1)
            destination_floor = start_floor
            while destination_floor == start_floor:
                destination_floor = random.randint(0, self.num_floors - 1)
            
            new_person = Person(start_floor, destination_floor, self)  # Truyền self (building) vào Person để dùng sim_time
            self.waiting_people[start_floor].append(new_person)
            self.last_spawn_time = self.sim_time

    def seasonality_spawn(self) -> None:
        """
        Docstring here
        """
        self.spawn_people(self.spawn_weight[self.current_day])
        
        if self.sim_time - self.day_timer > self.day_delay:
            self.day_timer = self.sim_time
            self.current_day = (self.current_day + 1) % 7

    def step(self, actions):
        """
        Docstring here
        """
        update_infos = []
        for i, action in enumerate(actions):
            update_infos.append(self.elevators[i].update(action))

        self.seasonality_spawn()

        self.sim_time += self.sim_step_size  # Tăng sim_time mỗi step
        
        self.delivered_people_count = sum(list(map(lambda x: x.delivered_people_count, self.elevators)))
        self.total_wait_time = sum(list(map(lambda x: x.total_wait_time, self.elevators)))
        self.total_travel_time = sum(list(map(lambda x: x.total_travel_time, self.elevators)))
        self.total_pickup_wait_time = self.total_wait_time - self.total_travel_time

        done = False
        if self.delivered_people_count >= self.need_to_carry:
            done = True

        return self.get_state(), update_infos, done, {}
