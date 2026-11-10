"""Contract tests for robot-feedback drawer motion; fixture truth stays outside."""
import numpy as np
from cross_episode_sim.control.sensor_drawer import SensorDrawerMotion


def test_observed_resistance_stops_without_fixture_limit_input():
    state=np.zeros(7)
    def tcp(q=None):
        T=np.eye(4);T[:3,3]=(state if q is None else q)[:3];return T
    class Planner:
        def plan(self,q,p,R):
            end=q.copy();end[:3]=p;return np.linspace(q,end,6)
    def command(q,grip):
        state[:]=q
        # A mocked physical stop, invisible to SensorDrawerMotion.
        state[0]=min(state[0],.10)
    motion=SensorDrawerMotion(joint_position=lambda:state.copy(),tcp_pose=tcp,
        fingers=lambda:np.zeros(2),command=command,
        planner_factory=lambda *args:Planner(),outward_normal=[1,0,0])
    result=motion.move_until_resistance(opening=True)
    assert result.completed and result.reason=='resisted_motion'
    assert abs(result.measured_travel-.10)<1e-9
    assert len(motion.events)<10


def test_failed_collision_plan_is_not_executed():
    commands=[]
    class Planner:
        def plan(self,*args):raise RuntimeError('blocked')
    motion=SensorDrawerMotion(joint_position=lambda:np.zeros(7),tcp_pose=lambda *args:np.eye(4),
        fingers=lambda:np.zeros(2),command=lambda *args:commands.append(args),
        planner_factory=lambda *args:Planner(),outward_normal=[1,0,0])
    result=motion.move_until_resistance(opening=False)
    assert not result.completed
    assert commands==[]
