from types import SimpleNamespace as NS
import numpy as np
import pytest

from cross_episode_sim.navigation.fixture_docking import (
    FixtureDocking, fixture_dock_candidates, observed_fixture_normal)
from cross_episode_sim.navigation.local_scan import ScanConfig


def test_vertical_handle_face_requires_no_horizontal_support():
    T=np.eye(4);T[:3,:3]=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    T[:3,3]=[0.,0.,.6]
    camera=NS(depth_m=np.ones((41,41)),valid_depth=np.ones((41,41),bool),
              robot_mask=None,K=np.array([[100.,0.,20.],[0.,100.,20.],[0.,0.,1.]]),T_base_optical=T)
    normal=observed_fixture_normal(NS(cameras={'nav':camera}),np.array([.96,0.,.6]))
    np.testing.assert_allclose(normal,[-1.,0.,0.],atol=1e-6)


def test_fixture_docks_face_diagonal_panel_and_remain_in_reach_band():
    n=np.array([-1.,-1.,0.])/np.sqrt(2);target=np.array([1.,2.,.6])
    for dock in fixture_dock_candidates(target,n):
        delta=target[:2]-dock[:2]
        assert .54<np.linalg.norm(delta)<.66
        np.testing.assert_allclose(delta/np.linalg.norm(delta),[np.cos(dock[2]),np.sin(dock[2])])


def test_missing_panel_returns_failure_without_uninitialized_checker(monkeypatch):
    import cross_episode_sim.navigation.fixture_docking as mod
    def missing(*args):raise ValueError('panel not visible')
    monkeypatch.setattr(mod,'observed_fixture_normal',missing)
    dock=FixtureDocking(NS(),NS());dock.reset(NS(position=np.array([1.,0.,.6])))
    monkeypatch.setattr(dock,'plan_observe',lambda *args:False)
    monkeypatch.setattr(dock,'plan_advance',lambda *args:False)
    packet=NS(timestamp=0.,arm=NS(position=np.array(ScanConfig().stowed_arm)))
    result=dock.step(packet,NS(accepted=True,T_map_base=np.eye(4)),0.)
    assert result['status']=='shape_route_unverified'
    assert result['forward']==0.


@pytest.mark.parametrize('task,target,expected',[
    ('set_table','bowl','scripts/demo_semantic_navigation.py'),
    ('clear_and_store','bowl','scripts/demo_semantic_navigation.py'),
    ('clear_and_store','handle on kitchen storage cabinet','scripts/demo_fixture_navigation.py'),
    ('prepare_and_serve_meal','handle on microwave','scripts/demo_fixture_navigation.py'),
    ('warm_two_pots','front left stove burner knob','scripts/demo_fixture_navigation.py')])
def test_only_added_fixture_targets_use_specialized_docking(monkeypatch,task,target,expected):
    from cross_episode_sim.evaluation.sensor_household import SensorHouseholdPipeline
    from cross_episode_sim.evaluation.sensor_set_table import SensorSetTablePipeline
    seen=[]
    monkeypatch.setattr(SensorSetTablePipeline,'stage',lambda self,name,cmd,*args:seen.append(cmd))
    pipeline=object.__new__(SensorHouseholdPipeline);pipeline.config={'task':task}
    pipeline.stage('navigate',['python','scripts/demo_semantic_navigation.py','--target',target],'report')
    assert seen[0][1]==expected


def test_generic_dock_fallback_reuses_checked_planner_only(monkeypatch):
    from cross_episode_sim.navigation.household_object_docking import HouseholdObjectDocking
    from cross_episode_sim.navigation.shaped_docking import ShapedDocking
    dock=HouseholdObjectDocking(NS(),NS());dock.reset(NS(position=np.array([1.,0.,.8])))
    monkeypatch.setattr(dock,'make_checker',lambda *args:NS(detail=None))
    def missing_face(self,*args):
        assert self.checker is not None
        self.terminal='shape_route_unverified'
        return dict(status=self.terminal,forward=0.,yaw=0.)
    monkeypatch.setattr(ShapedDocking,'step',missing_face)
    calls=[]
    monkeypatch.setattr(dock,'plan',lambda *args:calls.append(args) or True)
    result=dock.step(NS(arm=NS(position=np.zeros(7))),NS(accepted=True,T_map_base=np.eye(4)),0.)
    assert result['status']=='generic_object_dock_replan' and len(calls)==1
    assert dock.terminal is None
