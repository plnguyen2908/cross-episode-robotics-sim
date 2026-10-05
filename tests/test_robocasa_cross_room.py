"""Scene-adapter regressions without GPU planning or rollouts."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco

from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
from cross_episode_sim.manipulation.recovery import RecoveringManipulation


class CrossRoomAdapterTests(unittest.TestCase):
    def adapter(self, xml):
        check = CrossRoomManipulation.__new__(CrossRoomManipulation)
        check.model = mujoco.MjModel.from_xml_string(xml)
        check.data = mujoco.MjData(check.model)
        mujoco.mj_forward(check.model, check.data)
        check.bread_bids = set()
        return check

    def test_table_remains_in_arm_collision_world(self):
        check = CrossRoomManipulation.__new__(CrossRoomManipulation)
        check.table_gids = {3}
        with patch.object(RecoveringManipulation, 'kitchen_world_geoms', return_value=[1, 2]):
            self.assertEqual(check.kitchen_world_geoms(), [1, 2, 3])

    def test_ignore_registration_geometry_as_placement_surface(self):
        check = self.adapter('''<mujoco><worldbody><body name="counter">
          <geom type="box" pos="0 0 .90" size=".5 .3 .02"/>
          <geom type="box" pos="0 0 10" size=".01 .01 .01"/>
          </body></worldbody></mujoco>''')
        check.table_bids = {'counter': {check.model.body('counter').id}}
        boxes = check.surface_boxes('counter')
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0][2][2], .92)

    def test_backing_floor_does_not_override_room_labels(self):
        check = self.adapter('''<mujoco><worldbody>
          <body name="floor_kitchen"><geom type="box" pos="0 0 -.02" size="1 1 .02"/></body>
          <body name="floor_backing"><geom type="box" pos="0 0 -.14" size="1 1 .02"/></body>
          <body name="floor_dining"><geom type="box" pos="0 -2 -.02" size="1 1 .02"/></body>
          </worldbody></mujoco>''')
        with tempfile.TemporaryDirectory() as tmp:
            check.output = Path(tmp)
            nav = check._base_nav_map()
            names = set(nav.room_ids_to_name.values())
            self.assertEqual(names, {'floor_kitchen', 'floor_dining'})
            self.assertNotEqual(check.room_id([0, 0]), check.room_id([0, -2]))


if __name__ == '__main__':
    unittest.main()
