"""Two independently dynamic, qualified condiment instances in one episode."""
import copy
import xml.etree.ElementTree as ET
import numpy as np
from cross_episode_sim.fixtures.cabinet_transfer import CabinetTransfer, SHELF


def add_second_condiment(scene, object_name):
    tree = ET.parse(scene)
    root = tree.getroot()
    body = next(b for b in root.iter('body') if b.get('name') == object_name)
    parent = next(p for p in root.iter() if body in list(p))
    duplicate = copy.deepcopy(body)
    names = {n.get('name'): 'condiment2_' + n.get('name') for n in duplicate.iter() if n.get('name')}
    for node in duplicate.iter():
        for key, value in list(node.attrib.items()):
            if value in names:
                node.set(key, names[value])
    original_pos = np.fromstring(body.get('pos'), sep=' ')
    first_pos = original_pos + [-.13, 0, 0]
    second_pos = original_pos + [.13, 0, 0]
    body.set('pos', ' '.join(map(str, first_pos)))
    duplicate.set('pos', ' '.join(map(str, second_pos)))
    parent.append(duplicate)
    tree.write(scene)
    return names[object_name], first_pos.tolist(), second_pos.tolist()


class CondimentCollection(CabinetTransfer):
    video_filename = 'condiment_collection.mp4'

    def select_object(self, obj):
        super().select_object(obj)
        # Separate shelf destinations for the two objects, retaining collision
        # checks against the first object while manipulating the second.
        self.cabinet_slot_index = self.objects.index(obj)
        self.cabinet_slot_count = len(self.objects)

    def verify_collection(self):
        evidence = {}
        assignment = self.assignment()
        for obj in self.objects:
            self.select_object(obj)
            evidence[obj + ':supported_inside_cabinet'] = bool(
                assignment.get(obj) == SHELF and self.shelf_release_ready())
            evidence[obj + ':gripper_far'] = bool(
                np.linalg.norm(self.tcp()[:3, 3] - self.bread_pose()[:3, 3]) > .25)
        evidence['empty_hand'] = not self.attached and not self.holding_loaf
        evidence['cross_room'] = bool(self.room_id(self.initial_condiment_xy) != self.room_id([2.49, -.465]))
        return evidence

    def run_test(self):
        self.report.update(task='CondimentCollection',
            adaptation='two salt-shaker instances: dining table to native kitchen cabinet',
            native_goal='both condiments inside cabinet and gripper > 0.25 m from each',
            initial_state_extension='physically open initially closed cabinet',
            native_task_distribution_reproduced=False)
        self.initial_condiment_xy = self.bread_pose()[:2, 3].copy()
        steps = [dict(operation='cabinet_access', arguments=dict(opening=True))]
        steps.extend(dict(operation='transfer', arguments=dict(
            object_name=obj, source=self.dining, destination=SHELF)) for obj in self.objects)
        return 0 if self.run_composite_plan('CondimentCollection', steps, self.verify_collection) else 1
