from types import SimpleNamespace as NS
from cross_episode_sim.control.planner_cache import sensor_arm_planner
from cross_episode_sim.runtime.episode import EpisodeRuntime


def test_reuse_requires_same_geometry_and_sufficient_capacity(tmp_path):
    urdf=tmp_path/'robot.urdf';urdf.write_text('original geometry')
    calls=[];clears=[]
    def factory(*a,**kw):
        calls.append(kw)
        return NS(planner=NS(clear_scene_cache=lambda:clears.append(True)))
    kin=dict(urdf_path=str(urdf),asset_root_path=str(tmp_path))
    with EpisodeRuntime(artifact_root=tmp_path/'run').activate():
        a=sensor_arm_planner(kin,{'cuboid':100},factory)
        assert sensor_arm_planner(kin,{'cuboid':50},factory) is a
        assert len(calls)==1 and len(clears)==1
        b=sensor_arm_planner(kin,{'cuboid':200},factory)
        assert b is not a and len(calls)==2
        urdf.write_text('changed gripper geometry')
        assert sensor_arm_planner(kin,{'cuboid':50},factory) is not b
        assert len(calls)==3


def test_optimizer_clearance_does_not_reuse_default_solver(tmp_path):
    urdf=tmp_path/'robot.urdf';urdf.write_text('calibrated geometry')
    calls=[]
    def factory(*args,**kwargs):
        calls.append(kwargs)
        return NS(planner=NS(clear_scene_cache=lambda:None))
    kin=dict(urdf_path=str(urdf))
    with EpisodeRuntime(artifact_root=tmp_path/'run').activate():
        baseline=sensor_arm_planner(kin,{'cuboid':100},factory)
        fixture=sensor_arm_planner(kin,{'cuboid':100},factory,activation_distance=.012)
        assert fixture is not baseline
        assert 'activation_distance' not in calls[0]
        assert calls[1]['activation_distance']==.012
        assert sensor_arm_planner(kin,{'cuboid':80},factory) is baseline
