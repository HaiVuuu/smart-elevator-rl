import pygame
from constants import Color

class PersonView:
    def __init__(self, font):
        self.font = font

    def draw(self, screen, person, x, y):
        color = Color.BLUE
        if person.is_delivered:
            color = Color.GRAY
        elif person.is_in_elevator:
            color = Color.YELLOW
        pygame.draw.circle(screen, color, (x , y - 10), 5)
        text = self.font.render(str(person.destination_floor + 1), True, Color.BLACK)
        screen.blit(text, (x , y - 25))

class ElevatorView:
    def __init__(self, font):
        self.font = font
        self.person_view = PersonView(font)

    def draw(self, screen, elevator, x, y):
        elevator_rect = pygame.Rect(x, y, elevator.width, elevator.height)
        pygame.draw.rect(screen, Color.GRAY, elevator_rect)
        pygame.draw.rect(screen, Color.BLACK, elevator_rect, 2)

        door_x_left = x + 5
        door_x_right = x + elevator.width - elevator.door_width - 5
        pygame.draw.rect(screen, Color.BLACK, (door_x_left, y, elevator.door_width, elevator.height))
        pygame.draw.rect(screen, Color.BLACK, (door_x_right, y, elevator.door_width, elevator.height))
        
        for i, passenger in enumerate(elevator.passengers):
            self.person_view.draw(screen, passenger, x + 20, y + elevator.height - 10 - i*20)
            
        info_text = f"F:{elevator.floor+1} | {len(elevator.passengers)} P"
        text_surface = self.font.render(info_text, True, Color.BLACK)
        screen.blit(text_surface, (x, y - 20))

class BuildingView:
    def __init__(self, building, screen, font):
        self.building = building
        self.screen = screen
        self.font = font
        self.elevator_view = ElevatorView(font)
        self.person_view = PersonView(font)

    def draw(self):
        self.screen.fill(Color.WHITE)
        
        # draw floor
        for i in range(self.building.num_floors):
            y = (self.building.num_floors - 1 - i) * self.building.floor_height
            pygame.draw.line(self.screen, Color.BLACK, (0, y + self.building.floor_height), (self.screen.get_width(), y + self.building.floor_height), 2)
            floor_text = self.font.render(f"Floor {i + 1}", True, Color.BLACK)
            self.screen.blit(floor_text, (10, y + 10))

        # draw elevator
        for i, elevator in enumerate(self.building.elevators):
            elevator_x = 100 + i * (elevator.width + 25)
            elevator_y = (self.building.num_floors - 1 - elevator.floor) * self.building.floor_height + self.building.floor_height - elevator.height
            self.elevator_view.draw(self.screen, elevator, elevator_x, elevator_y)

        divider = sum(list(map(lambda x: x.width + 25, self.building.elevators)))

        # draw people
        for i, floor_list in enumerate(self.building.waiting_people):
            x = divider + 100
            y = (self.building.num_floors - 1 - i) * self.building.floor_height + self.building.floor_height - 10
            for j, person in enumerate(floor_list):
                self.person_view.draw(self.screen, person, x + j * 20, y)

        avg_wait_time = self.building.total_wait_time / self.building.delivered_people_count if self.building.delivered_people_count > 0 else 0
        avg_pickup_time = self.building.get_average_pickup_wait_time()
        avg_travel_time = self.building.get_average_travel_time()

        info_text = (
            f"Current day : {self.building.day_convert[self.building.current_day]} | "
            f"Delivered: {self.building.delivered_people_count} | "
            f"Avg Wait: {avg_wait_time:.2f}s | "
            f"Avg Pickup Wait: {avg_pickup_time:.2f}s | "
            f"Avg Travel: {avg_travel_time:.2f}s"
        )
        text_surface = self.font.render(info_text, True, Color.BLACK)
        self.screen.blit(text_surface, (10, self.screen.get_height() - 30))

        pygame.display.flip()
