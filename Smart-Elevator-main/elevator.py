from constants import ElevatorConfig

class Elevator:

    def __init__(
            self,
            building,
            width:int,
            height:int,
            door_width:int,
            capacity:int,
            run_delay:float
        ):

        self.floor = 0
        self.target_floor = 0
        self.passengers = []
        self.state = "idle"  # "idle", "moving_up", "moving_down"
        self.direction = 0
        self.capacity = capacity

        self.run_timer = building.sim_time  
        self.run_delay = run_delay

        self.width = width
        self.height = height
        self.door_width = door_width

        self.total_wait_time = 0
        self.total_travel_time = 0
        self.delivered_people_count = 0

        self.building = building

        
    def move_up(self) -> None:
        """
        Docstring here
        """
        if self.floor < self.building.num_floors-1:

            if self.building.sim_time - self.run_timer > self.run_delay:
                self.run_timer = self.building.sim_time
                self.floor += 1
                self.state = "moving_up"
                self.direction = 1


    def move_down(self) -> None:
        """
        Docstring here
        """
        if self.floor>0:

            if self.building.sim_time - self.run_timer > self.run_delay:
                self.run_timer = self.building.sim_time
                self.floor -= 1
                self.state = "moving_down"
                self.direction = -1


    def drop_off(self) -> int: # Return count of dropped off passengers
        """
        Handles dropping off passengers at the current floor.
        Returns the number of passengers dropped off.
        """
        passengers_to_remove = []
        wait_time = 0
        travel_time = 0
        for passenger in self.passengers:
            if passenger.destination_floor == self.floor: 
                passengers_to_remove.append(passenger)
                self.delivered_people_count += 1
                wait_time += passenger.get_wait_time()
                travel_time += passenger.get_travel_time()

        for p in passengers_to_remove:
            self.passengers.remove(p)

        self.total_wait_time += wait_time
        self.total_travel_time += travel_time

        return len(passengers_to_remove)


    def pick_up(self) -> int: # Return count of picked up passengers
        """
        Handles picking up passengers from the current floor.
        Returns the number of passengers picked up.
        """
        passengers_picked_up_count = 0
        floor_queue = self.building.waiting_people[self.floor]
        
        if self.direction == 0 and floor_queue and self.passengers:
            pass_dest = self.passengers[0].destination_floor
            if pass_dest > self.floor:
                self.direction = 1
            elif pass_dest < self.floor:
                self.direction = -1

        eligible_people = []
        for person in floor_queue:
            person_goes_up = person.destination_floor > self.floor
            person_goes_down = person.destination_floor < self.floor
            
            if (self.direction == 1 and person_goes_up) or \
               (self.direction == -1 and person_goes_down) or \
               self.direction == 0:
                eligible_people.append(person)

        for person in eligible_people:
            if len(self.passengers) < self.capacity:
                floor_queue.remove(person)
                self.passengers.append(person)
                person.is_in_elevator = True
                person.travel_start_time = self.building.sim_time
                passengers_picked_up_count += 1
            else:
                break

        return passengers_picked_up_count


    def update(self, action):
        """
        Updates the elevator state based on the action.
        Returns a dictionary with information about the update.
        """
        if action == 1:  # Move Up
            self.move_up()
        elif action == 2:  # Move Down
            self.move_down()
        else:  # action == 0 (Idle)
            self.state = 'idle'
            self.direction = 0

        passengers_dropped_off = self.drop_off()
        passengers_picked_up = self.pick_up()

        update_info = {
            'action': action,
            'passengers_dropped_off': passengers_dropped_off,
            'passengers_picked_up': passengers_picked_up,
            'is_idle': action == 0,
            'num_passengers': len(self.passengers),
        }
        return update_info




    def __repr__(self) -> str:
        """
        Docstring here
        """
        parsed_attribute = [ f"{key}={value}" for key,value in sorted(self.__dict__.items())]

        return f"Elevator({','.join(parsed_attribute)})"