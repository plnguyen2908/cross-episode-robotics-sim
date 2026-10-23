from .base import HouseholdCompositeTask
from .execution import call, fixture, transfer
from .room_route import route_plan


class WarmTwoPots(HouseholdCompositeTask):
    task_name = 'warm_two_pots'
    procedures = ('start_first_before_placing_second', 'place_both_then_start')


    def make_prompt(self):
        return (f"Keep the two cookware items on separate switched-on burners for at least {self.params['minimum_heating_seconds']:g} seconds. Finish with "
                'both burners off and both items remaining on the stove. Finish with an empty gripper.')

    def make_plan(self):
        place = [transfer(f'pot_{i:02d}', f'burner_{i}') for i in (1, 2)]
        start = [fixture('rotate_knob', f'burner_{i}_knob', state='on') for i in (1, 2)]
        plan = (place[0] + start[0] + place[1] + start[1]
                if self.procedure == 'start_first_before_placing_second'
                else place[0] + place[1] + start[0] + start[1])
        plan += [call('wait', duration_seconds=self.params['minimum_heating_seconds'])]
        for i in (1, 2):
            plan += fixture('rotate_knob', f'burner_{i}_knob', state='off')
        return route_plan(self.config, plan)
