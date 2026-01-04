

import numpy as np

class NormalAlgorithm:

    def predict(self,building,*args,**kwargs):
        """
        Docstring here
        """
        actions = []
        for elevator in building.elevators:
            if elevator.passengers:
                # If the elevator has passengers, the target is the destination of one of the passengers
                target_floor = min([p.destination_floor for p in elevator.passengers]) if elevator.direction == -1 else max([p.destination_floor for p in elevator.passengers])
                if elevator.floor < target_floor:
                    actions.append(1)  # Move up
                elif elevator.floor > target_floor:
                    actions.append(2)  # Move down
                else:
                    actions.append(0)  # Stop
            else:
                # If the elevator is empty, check for calls
                call_floors = [i for i, floor_list in enumerate(building.waiting_people) if floor_list]
                if call_floors:
                    # Go to the nearest floor with a waiting person
                    nearest_call = min(call_floors, key=lambda f: abs(f - elevator.floor))
                    if elevator.floor < nearest_call:
                        actions.append(1)
                    elif elevator.floor > nearest_call:
                        actions.append(2)
                    else:
                        actions.append(0)
                else:
                    actions.append(0) # Stay idle if no calls

        return np.array(actions)

