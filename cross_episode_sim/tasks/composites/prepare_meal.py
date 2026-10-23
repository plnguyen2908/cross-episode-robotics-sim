from .base import HouseholdCompositeTask
from .execution import call, fixture, transfer
from .room_route import route_plan


class PrepareAndServeMeal(HouseholdCompositeTask):
    task_name = 'prepare_and_serve_meal'
    procedures = ('heat_while_retrieving', 'retrieve_before_heating')


    def make_prompt(self):
        return (f"Run the microwave with the food container inside for at least {self.params['minimum_heating_seconds']:g} seconds. Bring it, "
                'the cold drink from the fridge to the dining table. '
                'Leave the microwave off and both appliance doors closed. Finish with an empty gripper.')

    def make_plan(self):
        retrieve = (fixture('open_door', 'fridge')
                    + transfer('drink_bottle_01', 'dining_table')
                    + fixture('close_door', 'fridge'))
        start = (fixture('open_door', 'microwave')
                 + transfer('food_container_01', 'microwave', 'interior')
                 + fixture('close_door', 'microwave')
                 + fixture('push_button', 'microwave_start'))
        end = ([call('wait', duration_seconds=self.params['minimum_heating_seconds'])]
               + fixture('push_button', 'microwave_stop')
               + fixture('open_door', 'microwave')
               + transfer('food_container_01', 'dining_table')
               + fixture('close_door', 'microwave'))
        return route_plan(self.config, (start + retrieve + end if self.procedure == 'heat_while_retrieving'
                else retrieve + start + end))
