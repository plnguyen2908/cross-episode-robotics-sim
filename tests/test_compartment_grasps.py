import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from cross_episode_sim.manipulation.compartment_grasps import shelf_lean_candidates


@pytest.mark.parametrize('yaw', [0., 90., 180.])
def test_shelf_lean_keeps_contact_point_and_top_down_cone(yaw):
    obj=np.eye(4);obj[:3,:3]=Rotation.from_euler('z',yaw,degrees=True).as_matrix();obj[:3,3]=[1,2,3]
    grasp=np.eye(4);grasp[:3,:3]=np.diag([1.,-1.,-1.]);grasp[:3,3]=[.02,0,.1]
    local=np.array([grasp]);before=local.copy()
    result=shelf_lean_candidates(local,obj,[0.,-1.])
    world=obj@result
    np.testing.assert_allclose(world[:,:3,3],np.repeat((obj@grasp)[None,:3,3],len(result),axis=0))
    assert np.all(world[:,2,2]<=-np.cos(np.radians(30))+1e-9)
    assert np.all(world[1:,1,2]>0)  # Wrist sits toward the exterior (-Y).
    np.testing.assert_array_equal(local,before)


def test_side_grasps_face_into_shelf_with_horizontal_closing():
    from cross_episode_sim.manipulation.compartment_grasps import shelf_side_candidates
    pose=np.eye(4);pose[:3,3]=[2.,3.,1.15]
    vertices=np.array([[2.+x,3.+y,1.15+z] for x in (-.025,.025)
                       for y in (-.025,.025) for z in (-.06,.06)])
    local=shelf_side_candidates(pose,vertices,[0.,-1.])
    world=pose@local
    assert len(world)==18
    np.testing.assert_allclose(world[:,2,2],0.,atol=1e-9)
    np.testing.assert_allclose(world[:,2,1],0.,atol=1e-9)
    assert np.all(world[:,1,2]>=np.cos(np.radians(15.))-1e-9)
    assert np.all(world[:,2,3]>vertices[:,2].min())
    assert np.all(world[:,2,3]<vertices[:,2].max())
    np.testing.assert_allclose(np.linalg.det(world[:,:3,:3]),1.)


def test_flexible_approaches_keep_top_and_side_available():
    from types import SimpleNamespace
    from cross_episode_sim.env.household import CompositeController
    c=SimpleNamespace(top_down_cone_degrees=30.)
    top=np.eye(4);top[:3,:3]=np.diag([1.,-1.,-1.])
    side=np.eye(4);side[:3,:3]=np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    assert CompositeController.annotation_approach_allowed(c,top)
    assert CompositeController.annotation_approach_allowed(c,side)
    assert not CompositeController.annotation_approach_allowed(c,np.eye(4))
    assert CompositeController.annotation_approach_penalty(c,top)<CompositeController.annotation_approach_penalty(c,side)
    candidates=[(0.,0,.04,side),(1.,1,.04,top),(2.,2,.04,top)]
    selected=CompositeController.select_annotation_candidates(c,candidates,1)
    assert [item[1] for item in selected]==[1,0]


def test_overhead_clearance_is_geometry_based():
    import mujoco
    from types import SimpleNamespace
    from cross_episode_sim.manipulation.compartment_grasps import overhead_clearance
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="unrelated_name" type="box" pos="0 0 .4" size=".2 .2 .01"/>
      <geom name="support" type="box" pos="0 0 0" size=".3 .3 .01"/>
      </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    c=SimpleNamespace(model=m,data=d,bread_bids=set(),robot_bids=set())
    vertices=np.array([[-.02,-.02,.02],[.02,.02,.2]])
    assert np.isclose(overhead_clearance(c,vertices),.19)
    assert np.isinf(overhead_clearance(c,vertices+np.array([1.,0.,0.])))


def test_rim_hypotheses_use_measured_top_and_keep_top_first():
    from types import SimpleNamespace
    from cross_episode_sim.manipulation.compartment_grasps import compartment_rim_candidates
    from cross_episode_sim.env.household import CompositeController
    obj=np.eye(4);obj[:3,3]=[2.,3.,1.]
    grasp=np.eye(4);grasp[:3,:3]=np.diag([1.,-1.,-1.])
    grasp[:3,3]=[.02,-.03,0.]
    local=np.array([grasp]);before=local.copy()
    vertices=np.array([[1.95,2.95,.98],[2.05,3.05,1.03]])
    world=obj@compartment_rim_candidates(local,obj,vertices,[0.,-1.])
    np.testing.assert_allclose(world[:,2,3],1.029)
    np.testing.assert_allclose(world[:,1,3],2.98)
    np.testing.assert_array_equal(local,before)
    c=SimpleNamespace(top_down_cone_degrees=30.,_compartment_rim_search=True)
    top=world[0];angled=world[-1]
    assert CompositeController.annotation_approach_allowed(c,angled)
    assert not CompositeController.annotation_approach_allowed(c,np.eye(4))
    selected=CompositeController.select_annotation_candidates(c,[(0.,0,.02,angled),(1.,1,.02,top)],1)
    assert [v[1] for v in selected]==[1,0]
    c._compartment_rim_search=False
    assert not CompositeController.annotation_approach_allowed(c,angled)


def test_compartment_rim_carry_preserves_grasp_orientation(monkeypatch):
    from cross_episode_sim.env.household import CompositeController
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    calls=[]
    c=object.__new__(CompositeController)
    c.object_name='container'
    c.object_info={'container':{'category':'food_container'}}
    c.tuck_vessel_for_navigation=lambda:calls.append('preserve')
    monkeypatch.setattr(BreakfastEpisode,'tuck_loaded_for_navigation',lambda self:calls.append('default'))
    c._compartment_rim_search=True
    c.tuck_loaded_for_navigation()
    c._compartment_rim_search=False
    c.tuck_loaded_for_navigation()
    assert calls==['preserve','default']


def test_pot_rim_fallback_accepts_tilt_without_allowing_upward_grasps():
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation
    from cross_episode_sim.env.household import CompositeController
    c = SimpleNamespace(top_down_cone_degrees=30, _rim_grasp_search=True)
    tilted = np.eye(4)
    tilted[:3, :3] = Rotation.from_euler('y', 140, degrees=True).as_matrix()
    assert CompositeController.annotation_approach_allowed(c, tilted)
    assert not CompositeController.annotation_approach_allowed(c, np.eye(4))
