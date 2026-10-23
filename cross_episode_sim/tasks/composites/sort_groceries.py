from .base import HouseholdCompositeTask
from .execution import roles, transfer
from .room_route import route_plan


class UnpackAndSortGroceries(HouseholdCompositeTask):
    task_name = 'unpack_and_sort_groceries'
    procedures = ('drinks_first', 'snacks_first')

    def make_prompt(self):
        return ('Put all packaged drinks in the purple basket and all packaged snacks '
                'in the orange basket. Leave both baskets in their initial locations. '
                'Leave unrelated objects where they are. Finish with an empty gripper.')

    def make_plan(self):
        groups = [('drink_bottle', 'purple_basket'), ('snack_box', 'orange_basket')]
        if self.procedure == 'snacks_first':
            groups.reverse()
        plan = [step for category, basket in groups
                for role in roles(category, self.params['object_counts'][category])
                for step in transfer(role, basket, 'interior')]
        return route_plan(self.config, plan)
