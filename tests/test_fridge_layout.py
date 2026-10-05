import unittest
from cross_episode_sim.controller.fridge_layout import FridgeSlot,choose_slot

class CapacityTest(unittest.TestCase):
    def setUp(self):
        self.right=FridgeSlot('right',2,0.,((0.,0.,0.),(.1,.1,.1)))
        self.left=FridgeSlot('left',3,0.,((0.,.5,0.),(.1,.6,.1)))
    def test_object_without_active_collision_mesh_is_empty(self):
        import mujoco
        from cross_episode_sim.controller.base import collision_mesh
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>')
        vertices,faces=collision_mesh(model,mujoco.MjData(model),lambda body:False)
        self.assertEqual(vertices.shape,(0,3))
        self.assertEqual(faces.shape,(0,3))
    def test_right_preferred_when_both_clear(self):
        self.assertEqual(choose_slot([self.left,self.right],[]),self.right)
    def test_full_right_uses_left(self):
        self.assertEqual(choose_slot([self.right,self.left],[self.right.bounds]),self.left)
    def test_right_only_rejects_full_right_even_with_empty_left(self):
        with self.assertRaisesRegex(RuntimeError,'left access is disabled'):
            choose_slot([self.right,self.left],[self.right.bounds],allowed_sides=('right',))
    def test_both_full_rejects(self):
        with self.assertRaisesRegex(RuntimeError,'Neither fridge'):
            choose_slot([self.right,self.left],[self.right.bounds,self.left.bounds])
    def test_clearance_accounts_for_nearby_objects(self):
        near=((.105,0.,0.),(.2,.1,.1))
        self.assertEqual(choose_slot([self.right,self.left],[near],clearance=.01),self.left)

if __name__=='__main__':unittest.main()
