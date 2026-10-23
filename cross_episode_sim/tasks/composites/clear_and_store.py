from .base import HouseholdCompositeTask
from .execution import fixture, roles, transfer
from .room_route import route_plan


class ClearAndStore(HouseholdCompositeTask):
    task_name = 'clear_and_store'
    procedures = ('finish_one_storage_at_a_time', 'open_both_then_store_then_close')


    def make_prompt(self):
        return ('Clear the bowls and cups from the dining table. Put the bowls in the '
                'kitchen cabinet and the cups in the drawer. Leave the storage compartments you used closed. Finish with an empty gripper.')

    def make_plan(self):
        groups = [('bowl', 'kitchen_cabinet', 'door'), ('cup', 'cup_drawer', 'drawer')]
        plan = []
        if self.procedure == 'open_both_then_store_then_close':
            for _, dest, kind in groups:
                plan += fixture('open_' + kind, dest)
        for category, dest, kind in groups:
            if self.procedure == 'finish_one_storage_at_a_time':
                plan += fixture('open_' + kind, dest)
            for obj in roles(category, self.params['object_counts'][category]):
                plan += transfer(obj, dest, 'interior')
            if self.procedure == 'finish_one_storage_at_a_time':
                plan += fixture('close_' + kind, dest)
        if self.procedure == 'open_both_then_store_then_close':
            for _, dest, kind in groups:
                plan += fixture('close_' + kind, dest)
        return route_plan(self.config, plan)
