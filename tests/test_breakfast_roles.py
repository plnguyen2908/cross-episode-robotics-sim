import unittest
import mujoco
import numpy as np

from cross_episode_sim.tasks.breakfast.scene import bind_roles, allocate_sites


class BreakfastRolesTest(unittest.TestCase):
    def rows(self):
        return [dict(source='molmo',asset=a,key='molmo__'+a,
                     placement_passes=1,cross_room_passes=0)
                for a in ['Book_1','Cup_1','Mug_1','Bowl_1']] + [
                    dict(source='robocasa',asset='lightwheel/honey_bottle/Honey001',
                         key='honey',placement_passes=1,cross_room_passes=1)]

    def test_roles_keep_asset_source_distinct_from_room(self):
        result=bind_roles(self.rows(),7)
        self.assertEqual(len(result),7)
        self.assertEqual({x['source_room'] for x in result},{'kitchen','living','dining'})
        self.assertEqual({x['role'] for x in result},set(x['role'] for x in bind_roles(self.rows(),7)))
        self.assertEqual(result[0]['source'],'molmo')
        self.assertEqual(result[0]['source_room'],'dining')

    def test_explicit_substitution_is_semantic_and_evidence_checked(self):
        rows=self.rows()
        result=bind_roles(rows,0,{'cup_one':'molmo__Mug_1'})
        self.assertEqual(next(x for x in result if x['role']=='cup_one')['asset'],'Mug_1')
        with self.assertRaises(ValueError):bind_roles(rows,0,{'cup_one':'molmo__Book_1'})
        rows[2]['placement_passes']=0
        with self.assertRaises(ValueError):bind_roles(rows,0,{'cup_one':'molmo__Mug_1'})

    def test_missing_role_does_not_silently_use_unrelated_object(self):
        with self.assertRaises(ValueError):bind_roles(self.rows()[:-1],0)
        with self.assertRaises(ValueError):bind_roles(self.rows(),0,{'typo':'honey'})

    def test_counter_hole_is_not_a_support(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="counter_main_main_group_main">
            <geom type="box" pos="-.6 0 .8" size=".4 .3 .02"/>
            <geom type="box" pos=".6 0 .8" size=".4 .3 .02"/>
          </body><body name="item" pos="0 0 1.5"><freejoint/>
          <geom type="box" size=".05 .05 .05"/></body>
          </worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        info={}
        allocate_sites(model,data,[dict(body='item',room='kitchen',field='position',
                                       info=info,preferred=.5)])
        self.assertGreater(abs(info['position'][0]),.25)

    def test_reserved_slot_cannot_be_reused(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="counter_main_main_group_main">
            <geom type="box" pos="0 0 .8" size=".3 .3 .02"/>
          </body><body name="item" pos="0 0 1.5"><freejoint/>
          <geom type="box" size=".05 .05 .05"/></body>
          </worldbody></mujoco>''')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        reserved={'kitchen':[(np.array([-1,-1,0]),np.array([1,1,2]))],
                  'dining':[],'living':[]}
        with self.assertRaises(ValueError):
            allocate_sites(model,data,[dict(body='item',room='kitchen',field='position',
                                           info={},preferred=.5)],reserved)


if __name__=='__main__':unittest.main()
