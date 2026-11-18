"""A receptacle outside the carry's final view is found by a checked look-around."""
import json,math
from types import SimpleNamespace
import numpy as np
import pytest
from cross_episode_sim.navigation import receptacle_approach as module
from cross_episode_sim.navigation.receptacle_approach import LOOK_STEP,approach_receptacle,look_headings

HALF_VIEW=math.atan(math.tan(math.radians(75)/2)*4/3)
wrap=lambda a:(a+math.pi)%(2*math.pi)-math.pi


def test_look_headings_cover_every_direction_once():
    yaw=.4;views=[yaw]+look_headings(yaw)
    assert len(views)==6 and all(abs(wrap(v-yaw))>1e-6 for v in views[1:])
    for direction in np.linspace(-math.pi,math.pi,721):
        # Each direction lies well inside one view, not only on an image edge.
        assert min(abs(wrap(direction-v)) for v in views)<=LOOK_STEP/2+1e-9<HALF_VIEW


class World:
    """Planar base with a receptacle that is in view only near one heading."""
    def __init__(self,monkeypatch,visible,blocked=lambda heading:False):
        self.yaw=0.;self.xy=np.zeros(2);self.data=SimpleNamespace(time=0.);self.model=None;self.turned=[]
        world=self;camera=SimpleNamespace(rgb=np.zeros((8,8,3),np.uint8),frame_id='navigation')
        self.observed_navigation_map=SimpleNamespace()
        self.sensors=SimpleNamespace(capture=lambda include_robot_mask:SimpleNamespace(cameras=dict(navigation=camera),
            timestamp=world.data.time,arm=SimpleNamespace(position=np.zeros(7))))
        def estimate(camera,timestamp):
            T=np.eye(4);c,s=math.cos(world.yaw),math.sin(world.yaw);T[:2,:2]=[[c,-s],[s,c]];T[:2,3]=world.xy
            return SimpleNamespace(accepted=True,T_map_base=T)
        self.mapper=SimpleNamespace(view=lambda:SimpleNamespace(update=estimate))
        class Base:
            def __init__(self,model,data,motion_guard):pass
            def step(self,speed,yaw):
                world.yaw+=yaw*.02;world.xy+=speed*.02*np.array([math.cos(world.yaw),math.sin(world.yaw)]);world.data.time+=.02
            def stationary(self):return True
        class Payload:
            def __init__(self,geometry,tool):pass
            def filter_packet(self,packet):return packet
            def checker(self,*a):return SimpleNamespace(remember_support=lambda pose:None)
        class Recorder:
            def __init__(self,*a):pass
            def __enter__(self):return self
            def __exit__(self,*a):return False
            def record_map(self,*a):pass
        class Docking:
            def __init__(self,grid,geometry):self.events=[];self.path=None
        def plan(docking,pose,q,candidates):
            goal=np.asarray(candidates[0],float)
            if blocked(wrap(goal[2])):return False
            world.turned.append(wrap(goal[2]));docking.path=[goal];return True
        for name,value in dict(RobotGeometry=lambda model:None,ObservedPayload=Payload,OnboardBaseActuator=Base,
                StateVideoRecorder=Recorder,ShapedDocking=Docking,plan_to_any_dock=plan,
                MolmoPointer=lambda:SimpleNamespace(point=lambda image,label:dict(points=[[4,4]] if abs(wrap(world.yaw-visible))<.1 else [])),
                LocalVLM=lambda device:SimpleNamespace(label_with_prompt=lambda images,prompt,max_new_tokens:json.dumps(dict(category='purple basket',description=''))),
                camera_points=lambda camera:(None,None,None),pixel_point=lambda camera,valid,pixel:np.array([1.,0.,.9]),
                placement_docks=lambda targets,pose,distances:[SimpleNamespace(pose=np.asarray(pose,float).copy())]).items():
            monkeypatch.setattr(module,name,value)
        monkeypatch.setattr('cross_episode_sim.navigation.observed_undock.departure_route',lambda pose,route,checker,keep_final_rotation:route)


def test_receptacle_beside_the_final_carry_view_is_found_by_turning(monkeypatch,tmp_path):
    world=World(monkeypatch,visible=2*LOOK_STEP)
    report=approach_receptacle(world,'purple basket',tmp_path/'approach',np.zeros((4,3)))
    assert report['success'] and np.allclose(report['look_headings'],[LOOK_STEP,2*LOOK_STEP])
    # The target is stored in the map frame of the view that saw it.
    assert np.allclose(report['target'],[math.cos(2*LOOK_STEP),math.sin(2*LOOK_STEP),.9],atol=.05)
    assert [e['event'] for e in report['route_events']]==['receptacle_look_turn']*2+['receptacle_approach_route']
    views=json.loads((tmp_path/'approach/observations.json').read_text())
    assert [v['view'] for v in views]==[2] and views[0]['accepted']


def test_visible_receptacle_needs_no_look(monkeypatch,tmp_path):
    world=World(monkeypatch,visible=0.)
    report=approach_receptacle(world,'purple basket',tmp_path/'approach',np.zeros((4,3)))
    assert report['look_headings']==[] and [e['event'] for e in report['route_events']]==['receptacle_approach_route']


def test_blocked_side_is_looked_at_from_the_other_side(monkeypatch,tmp_path):
    world=World(monkeypatch,visible=-LOOK_STEP,blocked=lambda heading:0<heading<math.pi-.01)
    report=approach_receptacle(world,'purple basket',tmp_path/'approach',np.zeros((4,3)))
    assert np.allclose(wrap(np.array(report['look_headings'])),[-LOOK_STEP])


def test_full_turn_without_the_receptacle_is_reported_not_guessed(monkeypatch,tmp_path):
    world=World(monkeypatch,visible=10.)
    with pytest.raises(RuntimeError,match='not visually confirmed'):
        approach_receptacle(world,'purple basket',tmp_path/'approach',np.zeros((4,3)))
    assert len(world.turned)==5 and (tmp_path/'approach/observations.json').exists()
