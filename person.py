import random
import time

class Person:
    """
    Represents a person waiting for an elevator.
    """
    def __init__(self, start_floor, destination_floor, building=None):

        self.start_floor = start_floor
        self.destination_floor = destination_floor
        self.is_in_elevator = False
        self.is_delivered = False
        self.x_offset = random.randint(-15, 15)
        
        self.building = building  # Thêm để truy cập sim_time
        self.spawn_time = building.sim_time if building else time.time()  # Sử dụng sim_time
        self.travel_start_time = None


    def get_wait_time(self):
        if self.building:
            return self.building.sim_time - self.spawn_time
        return time.time() - self.spawn_time


    def get_travel_time(self):
        if self.is_in_elevator:
            if self.building:
                return self.building.sim_time - self.travel_start_time
            return time.time() - self.travel_start_time

        return 0


    def draw(self, screen, x, y):

        color = Color.BLUE

        if self.is_delivered:

            color = Color.GRAY

        elif self.is_in_elevator:

            color = Color.YELLOW

        pygame.draw.circle(screen, color, (x , y - 10), 5)

        text = self.font.render(str(self.destination_floor + 1), True, Color.BLACK)

        screen.blit(text, (x , y - 25))

    
    def __repr__(self) -> str:
        """
        Docstring here
        """
        display_attribute = {
                'start_floor':1,
                'destination_floor':self.destination_floor,
                'spawn_time':self.spawn_time
                }
        parsed_attribute = [ f"{key}={value}" for key,value in sorted(display_attribute.items())]

        return f"Elevator({','.join(parsed_attribute)})"