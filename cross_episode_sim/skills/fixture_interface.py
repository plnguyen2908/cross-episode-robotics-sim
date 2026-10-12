"""Public, grounded atomic actions for the ten documented household families.

Controllers own geometry and contact control; this module owns argument schemas,
measured predicates and demo plans. A demo is never exposed as one atomic skill.
"""
import numpy as np

from cross_episode_sim.skills.atomic import BoundSkill
from cross_episode_sim.skills.interface import SkillBinding, SkillInterface, parameters, run_skill_episode

DEMOS = ('cross-room', 'drawer-pick-place', 'cabinet-door', 'drawer', 'stove-knob',
         'faucet', 'microwave-button', 'toaster-insertion', 'oven-rack', 'blender-lid')


def call(name, **arguments):
    return dict(operation=name, arguments=arguments)


def fixture_domain(c, demo):
    """Return the live interface, witness plan, setup and physical final goal."""
    if demo not in DEMOS:
        raise ValueError(f'Unsupported fixture demo: {demo}')
    if demo in ('cross-room', 'drawer-pick-place', 'toaster-insertion', 'blender-lid'):
        return object_domain(c, demo)
    bindings = {}
    empty = lambda: not bool(getattr(c, 'holding_loaf', False) or getattr(c, 'attached', False)
                             or getattr(c, 'articulating', False))
    def bind(name, execute, verify, schema, precondition, navigation=False):
        bindings[name] = SkillBinding(BoundSkill(c, name, execute, verify),
            execute.__doc__ or name.replace('_', ' '), schema, precondition, navigation)
    def at_goal():
        return bool(hasattr(c, '_fixture_dock') and np.linalg.norm(c.base_pose()[:2]-c._fixture_dock[:2]) < .025)
    def at_fixture():
        return at_goal() and getattr(c, '_fixture_dock_target', None)=='fixture'
    def dock():
        if demo == 'cabinet-door':return np.array([2.10,-1.20])
        if demo == 'drawer':return np.array([.50,-1.35])
        if demo == 'microwave-button':return np.array([3.41420304,-1.0])
        if demo == 'oven-rack':return np.array([3.7823,-1.10])
        xy = c.data.geom_xpos[c.handle_geom,:2].copy()+[0., -.82 if demo=='faucet' else -.48]
        return np.array([round(xy[0],2),round(xy[1]/.05)*.05])
    def navigate(target):
        """Tuck and navigate to the grounded fixture dock or retreat position."""
        c.tuck_for_navigation()
        c._precise_contact_boxes = set()
        start = c.base_pose().copy()
        goal = dock() if target=='fixture' else start[:2]-.25*np.array([np.cos(start[2]),np.sin(start[2])])
        c._fixture_dock = np.array([*goal,np.pi/2 if target=='fixture' else start[2]])
        c._fixture_dock_target = target
        c._faucet_retreat = target=='retreat' and demo in ('faucet','microwave-button')
        c._oven_retreat = target=='retreat' and demo=='oven-rack'
        try:c.task_navigate(goal,False,face=float(c._fixture_dock[2]))
        finally:
            c._faucet_retreat=False;c._oven_retreat=False;c._pickup_pre_nav_undock=0.
    bind('navigate',navigate,lambda target: at_goal() and bool(c.in_default_travel_posture(False)),
         parameters(target=('fixture','retreat')),lambda target:dict(empty_hand=empty()),True)
    plan=[call('navigate',target='fixture')]
    if demo in ('cabinet-door','drawer','stove-knob','faucet'):
        from cross_episode_sim.fixtures.faucet import water_on
        from cross_episode_sim.fixtures.stove_knob import burner_on
        name, states = {
            'cabinet-door':('set_door',('open','closed')),
            'drawer':('set_drawer',('open','closed')),
            'stove-knob':('twist_knob',('on','off')),
            'faucet':('turn_lever',('on','off'))}[demo]
        def satisfied(state):
            active = state==states[0]
            if demo=='cabinet-door':return bool(c.angle()>np.radians(50) if active else abs(c.angle())<np.radians(3))
            if demo=='drawer':return bool(c.angle()<-.17 if active else abs(c.angle())<.01)
            predicate=burner_on if demo=='stove-knob' else water_on
            return bool(predicate(c.angle()) if active else not predicate(c.angle()) and abs(c.angle())<.05)
        def articulate(state):
            """Grasp the fixture handle, change its passive joint, release and withdraw."""
            target = {'cabinet-door':np.pi/3,'drawer':-.20,'stove-knob':np.pi/3,'faucet':.48}[demo] if state==states[0] else 0.
            c.grasp_handle()
            (c.follow_slide if demo=='drawer' else c.follow_hinge)(target)
            c.release_handle()
        bind(name,articulate,lambda state:satisfied(state) and empty() and not bool(c.handle_contacts()),
             parameters(state=states),lambda state:dict(at_fixture=at_fixture(),empty_hand=empty()))
        plan += [call(name,state=s) for s in states]
        final=lambda:dict(fixture_restored=satisfied(states[1]),released=empty(),tucked=bool(c.in_default_travel_posture(False)))
    elif demo=='microwave-button':
        def press(button):
            """Press the named button with force-bearing contact, then withdraw."""
            c.embodiment.command_gripper(c.data,c.profile.gripper_close);c.tick(1.)
            c.rebuild_planner();c.press(button)
        bind('press_button',press,lambda button:bool(c.turned_on==(button=='start') and not c.contacts(button)[0]),
             parameters(button=('start','stop')),lambda button:dict(at_fixture=at_fixture(),empty_hand=empty()))
        plan += [call('press_button',button=b) for b in ('start','stop')]
        final=lambda:dict(microwave_off=not bool(c.turned_on),tucked=bool(c.in_default_travel_posture(False)))
    else:
        def joint(component):
            return float(c.data.qpos[c.model.jnt_qposadr[c.door_id if component=='door' else c.rack_id]])
        def move_component(component,state):
            c.select_component(component);c.grasp()
            c.follow((1.10 if component=='door' else .15) if state=='open' else 0.)
            c.release()
        def door(state):
            """Open or close the oven door, releasing its handle afterwards."""
            move_component('door',state)
        def rack(state):
            """Slide the upper rack out or in, releasing it afterwards."""
            move_component('rack',state)
        bind('set_door',door,lambda state:bool((joint('door')>1.05 if state=='open' else abs(joint('door'))<.05) and empty()),
             parameters(state=('open','closed')),lambda state:dict(at_fixture=at_fixture(),empty_hand=empty(),rack_in=abs(joint('rack'))<.01))
        bind('slide_rack',rack,lambda state:bool((joint('rack')>.13 if state=='open' else abs(joint('rack'))<.01) and empty()),
             parameters(state=('open','closed')),lambda state:dict(at_fixture=at_fixture(),empty_hand=empty(),door_open=joint('door')>1.05))
        plan += [call('set_door',state='open'),call('slide_rack',state='open'),call('slide_rack',state='closed'),call('set_door',state='closed')]
        final=lambda:dict(door_closed=abs(joint('door'))<.05,rack_in=abs(joint('rack'))<.01,released=empty(),tucked=bool(c.in_default_travel_posture(False)))
    plan.append(call('navigate',target='retreat'))
    def setup():
        c.tick(1.)
        if abs(c.angle())>.01:raise RuntimeError('Fixture demo requires its closed/off initial state')
        c.report.update(scope=f'Public atomic interface: {demo}',fixture_demo=demo)
    return SkillInterface(c,bindings),plan,setup,final


def object_domain(c,demo):
    """Separate navigation, qualified pickup and released placement/insertion."""
    from cross_episode_sim.fixtures.blender_lid import BLENDER
    bindings={}
    held=lambda:bool(getattr(c,'holding_loaf',False) and getattr(c,'attached',False) and len(c.contacts())>=2)
    empty=lambda:not bool(getattr(c,'holding_loaf',False) or getattr(c,'attached',False))
    def bind(name,execute,verify,schema,precondition,navigation=False):
        bindings[name]=SkillBinding(BoundSkill(c,name,execute,verify),execute.__doc__ or name.replace('_',' '),schema,precondition,navigation)
    def configure(source,destination):
        c.source,c.destination=source,destination;c.support_bids=c.table_bids[source]
        c.table_gids={g for g in range(c.model.ngeom) if c.model.geom_bodyid[g] in c.support_bids and (c.model.geom_contype[g] or c.model.geom_conaffinity[g])}
        c.initial_object_pose=c.bread_pose().copy();c.transfer_start=c.data.joint(c.object_joint).qpos.copy()
        c.preplanned_moves={};c._recovery_exhausted=False
    def navigate(target, dock='0'):
        """Reach the object's pickup dock or carry it to its destination dock."""
        if target=='source':
            if demo=='blender-lid':
                source=c.assignment()[c.object_name]
                configure(source,c.counter if source==BLENDER else BLENDER)
            c.prepare_pickup()
            if dock!='0':
                candidates=list(c.grasp_recovery_stances())
                pose=candidates[int(dock)-1]
                c.tuck_for_navigation()
                c.task_navigate(pose[:2],False,face=float(pose[2]))
                c.untuck_for_manipulation()
                c.planner=c.make_planner();c.arm_aids=c.actuator_ids(c.planner.names)
                c.load_world();c.preplanned_moves={}
            c._fixture_pick_prepared=c.object_name
        elif target=='destination':c.transport_payload()
        else:c.tuck_for_navigation()
        c._fixture_object_dock=c.base_pose().copy()
    def at_dock():
        return bool(hasattr(c,'_fixture_object_dock') and np.linalg.norm(c.base_pose()[:2]-c._fixture_object_dock[:2])<.025)
    nav_schema=parameters(target=('source','destination','retreat'))
    nav_schema['properties']['dock']=dict(type='string',enum=[str(i) for i in range(1 if demo=='drawer-pick-place' else 14)],default='0')
    bind('navigate',navigate,lambda target,dock='0':at_dock() and (target!='retreat' or bool(c.in_default_travel_posture(False))),nav_schema,
         lambda target,dock='0':dict(hand_state=held() if target=='destination' else empty(),dock_valid=dock=='0' or target=='source'),True)
    def pick(object):
        """Grasp and lift the named object at the prepared dock; verify its hold."""
        c._recovery_exhausted=True  # Alternative base docks require a public navigation call.
        c.pick_payload();c.attach_payload_to_planner();c._fixture_pick_prepared=None
    bind('pick',pick,lambda object:held(),parameters(object=(c.object_name,)),
         lambda object:dict(empty_hand=empty(),at_source=at_dock(),prepared=getattr(c,'_fixture_pick_prepared',None)==object))
    def place(object,destination):
        """Place, release and withdraw the named object on its grounded support."""
        if demo!='cross-room':c.transport_payload()
        c.place_payload()
    if demo!='toaster-insertion':
        bind('place',place,lambda object,destination:bool(empty() and c.assignment()[object]==c.destination),
             parameters(object=(c.object_name,),destination=('support',)),lambda object,destination:dict(held=held(),at_dock=at_dock()))
    plan=[call('navigate',target='source'),call('pick',object=c.object_name)]
    if demo=='cross-room':plan.append(call('navigate',target='destination'))
    if demo=='toaster-insertion':
        def insert(object,fixture):
            """Align the held bread, insert into the slot, release and withdraw."""
            c.transport_payload();c.place_payload()
        def inserted():
            s=c.insertion_state()
            return bool(empty() and s['slot_floor_contact'] and s['footprint_inside'] and s['insertion_depth_m']>.04 and s['gripper_distance_m']>.15)
        bind('insert',insert,lambda object,fixture:inserted(),parameters(object=(c.object_name,),fixture=('toaster_slot',)),lambda object,fixture:dict(held=held(),at_dock=at_dock()))
        plan.append(call('insert',object=c.object_name,fixture='toaster_slot'))
        final=lambda:dict(inserted=inserted(),tucked=bool(c.in_default_travel_posture(False)))
    elif demo=='blender-lid':
        def open_lid(fixture):
            """Lift the removable lid off its fixture and retain a verified grasp."""
            pick(c.object_name)
        def close_lid(fixture):
            """Reseat the held lid, release it and withdraw from the fixture."""
            c.transport_payload();c.place_payload()
        bind('open_lid',open_lid,lambda fixture:held() and not bool(c.state()['lid_on_blender']),parameters(fixture=('blender',)),
             lambda fixture:dict(empty_hand=empty(),at_dock=at_dock(),lid_seated=bool(c.state()['lid_on_blender']),prepared=getattr(c,'_fixture_pick_prepared',None)==c.object_name))
        bind('close_lid',close_lid,lambda fixture:bool(empty() and c.state()['lid_on_blender'] and c.assignment()[c.object_name]==BLENDER),parameters(fixture=('blender',)),
             lambda fixture:dict(held=held(),at_dock=at_dock()))
        plan=[call('navigate',target='source',dock='1'),call('open_lid',fixture='blender'),call('place',object=c.object_name,destination='support'),
              call('navigate',target='source'),call('pick',object=c.object_name),call('close_lid',fixture='blender')]
        final=lambda:dict(lid_seated=bool(c.state()['lid_on_blender']),released=empty(),tucked=bool(c.in_default_travel_posture(False)))
    else:
        plan.append(call('place',object=c.object_name,destination='support'))
        plan.append(call('navigate',target='retreat'))
        def final():
            result=dict(on_destination=bool(c.assignment()[c.object_name]==c.destination),released=empty(),tucked=bool(c.in_default_travel_posture(False)))
            if demo=='cross-room':result['cross_room_navigation_completed']=bool(c.report.get('cross_room_navigation_completed'))
            else:result['base_remained_parked']=bool(np.linalg.norm(c.base_pose()[:2]-c._fixture_start_base[:2])<.01)
            return result
    def setup():
        c.tick(1.)
        c._fixture_start_base=c.base_pose().copy()
        if demo=='cross-room':c.source,c.destination=c.receptacles
        if demo=='blender-lid':
            if not c.state()['lid_on_blender'] or c.assignment()[c.object_name]!=BLENDER:
                raise RuntimeError('Lid demo must start physically seated')
            c.closed_pose=c.bread_pose().copy()
        configure(c.source,c.destination)
        c.report.update(scope=f'Public atomic interface: {demo}',fixture_demo=demo)
    return SkillInterface(c,bindings),plan,setup,final


def run_fixture_demo(controller,demo):
    api,plan,setup,goal=fixture_domain(controller,demo)
    recovery=FixtureDockRecovery() if demo in ('cross-room','blender-lid','toaster-insertion') else None
    return 0 if run_skill_episode(controller,api,plan,goal,demo,setup=setup,recover=recovery) else 1


class FixtureDockRecovery:
    """Oracle policy: propose explicit navigation calls after grasp failures."""
    def __init__(self):
        self.next_dock=0
        self.action=None

    def __call__(self,step,error):
        name=step['operation']
        if 'precondition_failed' in str(error):return None
        if name not in ('pick','open_lid','navigate'):return None
        if name=='navigate' and step['arguments']['target']!='source':return None
        if name!='navigate':
            key=(name,tuple(sorted(step['arguments'].items())))
            if self.action!=key:self.next_dock=0
            self.action=key
        self.next_dock+=1
        if self.next_dock>13:return None
        retry=[call('navigate',target='source',dock=str(self.next_dock))]
        # A failed navigation already has its pickup next in the plan.
        if name!='navigate':retry.append(dict(operation=name,arguments=dict(step['arguments'])))
        return retry


def fixture_skill_interface(controller, demo):
    """Initialize a fresh fixture controller for an external oracle/VLM planner.

    Call once after scene construction. Initialization settles the authored
    scene and binds source/goal metadata; robot actions use execute_skill.
    """
    api, _, setup, _ = fixture_domain(controller, demo)
    setup()
    controller.skill_interface = api
    return api
