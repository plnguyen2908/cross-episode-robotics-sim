from .base import HouseholdCompositeTask
from .execution import transfer
from .room_route import route_plan


class SetTable(HouseholdCompositeTask):
    task_name = 'set_table'
    procedures = ('by_place_setting', 'by_object_category')

    def make_prompt(self):
        return (f"Set the dining table for {self.params['people']} people. Give each place "
                'setting one bowl and one cup. Leave extra items where they are. Finish with an empty gripper.')

    def make_plan(self):
        n = self.params['people']
        pairs = ([(c, i) for i in range(n) for c in ('bowl', 'cup')]
                 if self.procedure == 'by_place_setting' else
                 [(c, i) for c in ('bowl', 'cup') for i in range(n)])
        return route_plan(self.config, [step for c, i in pairs for step in
                transfer(f'{c}_{i+1:02d}', 'dining_table', f'setting_{i+1:02d}_{c}')])
