"""Native closed-storage authoring and adapters for breakfast retrieval.

The adapters share the live model, data, recorder and planner with breakfast;
they never instantiate/reset a second simulator between operations.
"""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from cross_episode_sim.tasks.breakfast.scene import bounds, support_bounds
from cross_episode_sim.manipulation.edge_access import object_bodies
from cross_episode_sim.fixtures.cabinet_transfer import CabinetTransfer, SHELF, SHELF_GEOM
from cross_episode_sim.fixtures.cabinet_door import CabinetDoorTest, DOOR, HINGE, restore_native_door
from cross_episode_sim.fixtures.drawer import DrawerTest, DRAWER


def prepare_storage(manifest):
    scene=Path(manifest['scene_xml'])
    wanted={i['initial_source'] for i in manifest['bindings']} & {'drawer','cabinet'}
    if not wanted:return
    recording=Path(manifest['bindings'][0]['recording'])
    fixtures={'cabinet':(DOOR,HINGE,SHELF),'drawer':(DRAWER,DrawerTest.fixture_joint,DRAWER)}
    for kind in sorted(wanted):
        root,joint,support=fixtures[kind]
        restore_native_door(recording,scene,root)
        manifest['supports'][kind]=support;manifest['outward'][kind]=[0.,-1.]
        manifest['dynamic_fixtures'].append(root)
        manifest['storage'][kind]=dict(root=root,joint=joint,support=support)
    tree=ET.parse(scene);model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    rng=np.random.default_rng(manifest['seed']+4701)
    reserved=[]
    for info in manifest['bindings']:
        kind=info['initial_source']
        if kind not in wanted:continue
        support=manifest['supports'][kind];lo,hi=support_bounds(model,data,support)
        olo,ohi=bounds(model,data,info['body']);origin=data.body(info['body']).xpos.copy()
        lower,upper=olo-origin,ohi-origin
        xmin,xmax=lo[0]+.012-lower[0],hi[0]-.012-upper[0]
        if xmin>=xmax:raise ValueError(f'{info["role"]} does not fit closed {kind}')
        jid=model.body_jntadr[model.body(info['body']).id];address=model.jnt_qposadr[jid]
        accepted=False
        for _ in range(160):
            point=np.array([rng.uniform(xmin,xmax),lo[1]+.02-lower[1],hi[2]+.002-lower[2]])
            a,b=lower+point,upper+point
            if b[1]>hi[1]-.01:break
            if any(k==kind and np.all(a[:2]-.025<ob[:2]) and np.all(b[:2]+.025>oa[:2]) for k,oa,ob in reserved):continue
            data.qpos[address:address+3]=point;mujoco.mj_forward(model,data)
            owners=object_bodies(model,info['body']);support_ids=object_bodies(model,support)
            blocked=False
            for c in data.contact:
                b1,b2=model.geom_bodyid[[c.geom1,c.geom2]]
                if (b1 in owners)==(b2 in owners):continue
                other=b2 if b1 in owners else b1
                if c.dist<-.001 and other not in support_ids:blocked=True;break
            if blocked:continue
            info['source']=kind;info['position']=point.tolist()
            tree.find(f".//body[@name='{info['body']}']").set('pos',' '.join(map(str,point)))
            reserved.append((kind,a,b));accepted=True;break
        if not accepted:raise ValueError(f'No closed-storage collision-free fit for {info["role"]}/{kind}')
    tree.write(scene)
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    for _ in range(round(2/model.opt.timestep)):mujoco.mj_step(model,data)
    for kind,spec in manifest['storage'].items():
        if abs(float(data.joint(spec['joint']).qpos[0]))>.01:
            raise ValueError(f'Population pushed {kind} open; refusing closed-start scene')
    for info in manifest['bindings']:
        owners=object_bodies(model,info['body']);support=object_bodies(model,manifest['supports'][info['source']])
        supported=False
        for c in data.contact:
            a,b=model.geom_bodyid[[c.geom1,c.geom2]]
            if (a in owners)==(b in owners):continue
            other=b if a in owners else a
            nz=-c.frame[2] if a in owners else c.frame[2]
            if other in support and nz>.7 and c.dist<=.002:supported=True
            if other not in support and c.dist<-.001:
                raise ValueError(f'Closed-storage population collision: {info["role"]}')
        if not supported:raise ValueError(f'Unsupported storage population: {info["role"]}')
        node=tree.find(f".//body[@name='{info['body']}']")
        info['position']=data.body(info['body']).xpos.tolist()
        info['initial_quaternion']=data.body(info['body']).xquat.tolist()
        node.set('pos',' '.join(map(str,info['position'])))
        node.set('quat',' '.join(map(str,info['initial_quaternion'])))
    tree.write(scene)
    manifest['storage_population_validation']='closed, separated, physically supported; retrieval unqualified'


def storage_closed(controller):
    return all(abs(float(controller.data.joint(s['joint']).qpos[0])) <
               (.01 if k=='drawer' else np.radians(3)) for k,s in controller.manifest['storage'].items())


def shelf_exit_waypoints(start):
    """Clear the countertop lip before raising the wrist inside the opening."""
    poses=[]
    for retreat,rise in ((.045,.003),(.09,.025)):
        pose=start.copy();pose[1,3]-=retreat;pose[2,3]+=rise;poses.append(pose)
    extracted=poses[-1].copy();extracted[1,3]=min(extracted[1,3]-.28,-.82)
    raised=extracted.copy();raised[2,3]=start[2,3]+.14
    return poses+[extracted,raised]


def drawer_pickup_stances(controller):
    """Prefer the achieved opening stance; derive alternatives from the live drawer."""
    current=controller.base_pose().copy()
    target=controller.bread_pose()[:3,3]
    direction=target[:2]-current[:2]
    heading=float(np.arctan2(direction[1],direction[0]))
    error=float(np.arctan2(np.sin(heading-current[2]),np.cos(heading-current[2])))
    # A coarse candidate filter only; annotation IK still decides arm reach.
    if np.linalg.norm(direction)<=.8 and abs(error)<np.radians(60):
        yield current
    low,_=support_bounds(controller.model,controller.data,DRAWER)
    for distance,lateral in ((.55,0.),(.65,0.),(.55,.10),(.55,-.10),(.45,0.)):
        pose=np.array([target[0]+lateral,low[1]-distance,np.pi/2])
        if np.linalg.norm(pose-current)>.005:yield pose


def approach_drawer_pickup(controller):
    rejected=[]
    chosen = getattr(controller, '_trial_storage_dock', None)
    docks = [chosen] if chosen is not None else drawer_pickup_stances(controller)
    blocked=[]
    for dock in docks:
        try:
            route=controller.plan_route(dock[:2],False,face=float(dock[2]))
        except RuntimeError as exc:
            rejected.append(dict(dock=dock.tolist(),reason=str(exc)))
            blocked.append(dock);continue
        controller.record(drawer_pickup_dock=dock.tolist(),
            reused_opening_stance=bool(np.linalg.norm(dock-controller.base_pose())<.005),
            rejected_drawer_docks=rejected)
        # Keep the swept-checked route and its undocking metadata together.
        controller._accepted_route=route
        controller.task_navigate(dock[:2],False,face=float(dock[2]))
        return
    # Only when no dock has a direct route, reach one through another stance.
    for dock in blocked:
        if via_drawer_pickup_stance(controller,dock,rejected):
            return
    raise RuntimeError(f'No swept-clear empty-hand drawer pickup dock: {rejected}')


def via_drawer_pickup_stance(controller,dock,rejected):
    """Reach a dock whose direct route is blocked through another pickup stance.

    Beside the open drawer and the wall there is no room to turn straight into
    the closest stance from the opening stance; it is reachable after first
    stepping to a neighbouring stance. Each leg keeps the full swept checks.
    """
    current=controller.base_pose().copy()
    vias=sorted((s for s in drawer_pickup_stances(controller)
                 if np.linalg.norm(s-current)>.005 and np.linalg.norm(s-dock)>.005),
                key=lambda s:np.linalg.norm(s[:2]-dock[:2]))
    for via in vias:
        try:
            route=controller.plan_route(via[:2],False,face=float(via[2]))
        except RuntimeError:
            continue
        controller._accepted_route=route
        controller.task_navigate(via[:2],False,face=float(via[2]))
        # Once the base has moved, a failed second leg is a failed trial; the
        # storage pickup skill restores the checkpoint before the next dock.
        route=controller.plan_route(dock[:2],False,face=float(dock[2]))
        controller.record(drawer_pickup_dock=dock.tolist(),drawer_pickup_via=via.tolist(),
            reused_opening_stance=False,rejected_drawer_docks=rejected)
        controller._accepted_route=route
        controller.task_navigate(dock[:2],False,face=float(dock[2]))
        return True
    return False


def cabinet_pickup_stances(controller):
    target = controller.bread_pose()[:3, 3]
    low, _ = support_bounds(controller.model, controller.data, SHELF)
    for distance, lateral in ((.50,-.28),(.55,-.28),(.45,-.28),(.55,-.38),
                              (.75,0.),(.85,0.),(.55,0.),(.65,0.)):
        yield np.array([target[0]+lateral, low[1]-distance, np.pi/2])


def drawer_departure_budget(controller, stage, target):
    """Existing sub-0.1 mm payload/wall contacts may separate during vertical lift."""
    if stage!='lift bread' or not getattr(controller,'holding_loaf',False):return None
    start=controller.tcp()
    if (target[2,3]-start[2,3]<.03 or np.linalg.norm(target[:2,3]-start[:2,3])>.005
            or np.linalg.norm(target[:3,:3]-start[:3,:3])>.02):return None
    fixture=controller.descendants(DRAWER)
    depths={}
    for contact in controller.data.contact:
        a,b=controller.model.geom_bodyid[[contact.geom1,contact.geom2]]
        if not ((a in controller.bread_bids and b in fixture) or
                (b in controller.bread_bids and a in fixture)):continue
        key=tuple(sorted((int(contact.geom1),int(contact.geom2))))
        depths[key]=max(depths.get(key,0.),-float(contact.dist))
    depths={k:v for k,v in depths.items() if 0 < v <= .0001}
    if not depths:return None
    return dict(depths=depths,start_z=float(controller.bread_pose()[2,3]))


def drawer_departure_contact_view(controller,data,carrying):
    budget=getattr(controller,'_drawer_departure_budget',None)
    if not carrying or not budget:return data
    dz=float(data.body(controller.object_name).xpos[2])-budget['start_z']
    if not 0 <= dz <= .02:return data
    depths={}
    for contact in data.contact:
        key=tuple(sorted((int(contact.geom1),int(contact.geom2))))
        depths[key]=max(depths.get(key,0.),-float(contact.dist))
    previous=budget.setdefault('previous_depths',dict(budget['depths']))
    separating={key for key,initial in previous.items()
                if depths.get(key,0.)<=initial+1e-6}
    for key in separating:previous[key]=min(previous[key],depths.get(key,0.))
    # Pass all other contacts unchanged through the existing collision policy.
    keep=np.array([tuple(sorted((int(c.geom1),int(c.geom2)))) not in separating
                   for c in data.contact],dtype=bool)
    class Contacts:
        """The kept contacts, iterable per contact and as field arrays like MjContactList."""
        def __init__(self):
            self._items=[c for c,k in zip(data.contact,keep) if k]
        def __iter__(self):return iter(self._items)
        def __len__(self):return len(self._items)
        def __getitem__(self,index):return self._items[index]
        def __getattr__(self,name):return np.asarray(getattr(data.contact,name))[keep]
    class ContactView:
        def __init__(self):
            self.contact=Contacts()
            self.ncon=len(self.contact)
        def __getattr__(self,name):return getattr(data,name)
    return ContactView()


def retrieve_from_storage(controller, info):
    # Import lazily to avoid a module cycle.
    from cross_episode_sim.tasks.breakfast.gather import GatherBreakfast
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    from cross_episode_sim.tasks.breakfast.office import OfficeBreakfastEpisode
    from cross_episode_sim.fixtures.drawer_loop import DrawerLoop
    from cross_episode_sim.fixtures.drawer_pick_place import DrawerPickPlace
    from cross_episode_sim.controller.manipulation import TableReorder

    class CabinetBreakfast(CabinetTransfer, GatherBreakfast):
        shelf_diagonal_grasps=True
        prepare_destination=OfficeBreakfastEpisode.prepare_destination
        redock_loaded_for_placement=BreakfastEpisode.redock_loaded_for_placement
        place_payload=TableReorder.place_payload
        object_label=OfficeBreakfastEpisode.object_label
        # Keep the scene-wide recorder/content monitor during fixture operation.
        tick=GatherBreakfast.tick
        before_step=GatherBreakfast.before_step
        render_video_frame=GatherBreakfast.render_video_frame

        def pick_payload(self):
            from cross_episode_sim.skills.fixtures import StoragePickupSkill
            return StoragePickupSkill(self, cabinet_pickup_stances(self)).run()

        deliver_payload=BreakfastEpisode.deliver_payload

        def execute_transfer(self):
            """Re-pick with another rim grasp if the carried vessel drops.

            A rim grasp can pass the two-second hold and still slip out while the
            base moves. Delivery retries restart from the post-pickup checkpoint
            with that same grasp, so a drop rolls back to before the pickup and
            excludes the grasp. The exclusion list lives outside the controller
            because rollback restores every controller attribute.
            """
            from cross_episode_sim.skills.atomic import CallbackSkill
            dropped=set()

            def execute(attempt):
                self._dropped_grasp_variants=set(dropped)
                try:
                    return CabinetTransfer.execute_transfer(self)
                except RuntimeError as exc:
                    variant=getattr(self,'selected_annotation_variant',None)
                    if 'Object dropped after clearing pickup support' in str(exc) and variant is not None:
                        dropped.add(variant)
                    raise

            def rebuild():
                self.planner=self.make_planner();self.arm_aids=self.actuator_ids(self.planner.names)
                self.load_world()

            return CallbackSkill(self,'storage_transfer',range(3),execute,
                                 verify=lambda:bool(self.report.get('success')),rebuild=rebuild).run()

        def transport_payload(self):
            info=self.object_info[self.object_name]
            point=np.asarray(info['destination_position'])
            base=self.base_pose()
            nearby=np.linalg.norm(point[:2]-base[:2]) <= max(
                offset[0] for offset in self.profile.manipulation_offsets)+.10
            if (self.source==SHELF and info['destination']=='kitchen' and nearby
                    and not getattr(self, '_redock_index', 0)
                    and self.room_id(base[:2])==self.room_id(point[:2])):
                # Try the adjacent counter from the extracted pose. cuRobo
                # still checks the loaded trajectory; placement retains its
                # normal alternate-spot/redock recovery if this dock cannot reach.
                self._placement_pick_yaw=float(base[2])
                self.review_phase=f"PLACE {info['role']} ON ADJACENT COUNTER"
                self.record(cabinet_counter_policy='try current extraction dock before loaded navigation')
                self.prepare_destination()
                return
            return BreakfastEpisode.transport_payload(self)

        def move(self,stage,pose):
            if self.source==SHELF and stage=='lift bread vertically':
                for waypoint in shelf_exit_waypoints(self.tcp())[:2]:
                    super().move(stage,waypoint)
                return
            if self.source==SHELF and stage=='lift bread':
                extracted=self.tcp().copy();extracted[1,3]=min(extracted[1,3]-.28,-.82)
                raised=extracted.copy()
                raised[2,3]=self.pickup_start_height+.14+(extracted[2,3]-self.bread_pose()[2,3])
                for label,target in (('extract held vessel from cabinet',extracted),
                                     ('raise extracted vessel',raised)):
                    try:super().move(label,target)
                    except RuntimeError as exc:
                        if not str(exc).startswith(('cuRobo failed to plan','cuRobo could not plan','No nearby placement IK')):
                            raise
                        # Preflight used this same local contact path. If the
                        # free-space solver rejects its IK branch, recheck it
                        # from the measured hold before executing any commands.
                        path=self.plan_contact_path(label,target)
                        self.record(cabinet_extraction_fallback='actual_mesh_checked_contact_path',
                                    planner_error=str(exc))
                        self.mesh_contact_move(label,target,path=path)
                return
            return super().move(stage,pose)

        def validate_grasp_lift(self,joints,aperture):
            from scipy.spatial.transform import Rotation
            live=self.data;held=getattr(self,'holding_loaf',False)
            probe=mujoco.MjData(self.model);probe.qpos[:]=live.qpos
            probe.qvel[:]=live.qvel;probe.ctrl[:]=live.ctrl;probe.time=live.time
            addresses=[self.model.jnt_qposadr[self.model.joint('robot_0/'+n).id]
                       for n in self.planner.names]
            probe.qpos[addresses]=joints
            self.embodiment.probe_aperture(self.model,probe,aperture)
            mujoco.mj_forward(self.model,probe)
            try:
                self.data=probe;self.holding_loaf=True
                relative=np.linalg.inv(self.tcp())@self.bread_pose()
                for index,pose in enumerate(shelf_exit_waypoints(self.tcp())):
                    path=self.plan_contact_path(f'cabinet extraction preflight {index+1}',pose)
                    payload=pose@relative
                    probe.qpos[addresses]=path[-1]
                    probe.joint(self.object_joint).qpos[:3]=payload[:3,3]
                    probe.joint(self.object_joint).qpos[3:7]=Rotation.from_matrix(payload[:3,:3]).as_quat(scalar_first=True)
                    mujoco.mj_forward(self.model,probe)
            finally:
                self.data=live;self.holding_loaf=held

        def prepare_shelf_grasps(self):
            from scipy.spatial.transform import Rotation
            from cross_episode_sim.fixtures.cabinet_transfer import front_shelf_approach
            originals=self.local_annotations.copy()
            super().prepare_shelf_grasps()
            # Reorient existing rim-contact hypotheses around the vessel rather
            # than forcing a whole-bowl pinch. These rotated poses are UNVERIFIED
            # until the usual actual-mesh, IK and physical lift checks pass.
            object_pose=self.bread_pose();inverse=np.linalg.inv(object_pose)
            rim=[]
            for local in originals:
                world=object_pose@local
                for yaw in np.linspace(0,2*np.pi,24,endpoint=False):
                    rotation=Rotation.from_euler('z',yaw).as_matrix()
                    pose=world.copy();pose[:3,:3]=rotation@world[:3,:3]
                    pose[:3,3]=object_pose[:3,3]+rotation@(world[:3,3]-object_pose[:3,3])
                    if front_shelf_approach(pose,True) and pose[1,3]<object_pose[1,3]:
                        rim.append(inverse@pose)
            if rim:
                self.local_annotations=np.asarray(rim)
                # Center the rim farther inside the pads. The unshifted rim
                # poses could pass a short lift but slipped during wrist motion.
                # These remain hypotheses: all width/mesh/IK checks still apply.
                self.local_annotations[:,:3,3]+=.015*self.local_annotations[:,:3,2]
                self.annotation_path=self.output/'cabinet_rotated_rim_hypotheses.npz'
                np.savez_compressed(self.annotation_path,transforms=self.local_annotations)
                self.args.annotation_source='rotated_rim_hypotheses'
                self.record(cabinet_rotated_rim_candidates=len(rim),physically_qualified=False,
                            rim_grasp_insertion_offset_m=.015)
            self.annotation_candidate_budget=96
            # Grasps that dropped the vessel in an earlier carry; regenerated
            # from the same restored state, their indices are unchanged.
            self.physically_rejected_annotation_variants=(
                set(getattr(self,'physically_rejected_annotation_variants',set()))
                |set(getattr(self,'_dropped_grasp_variants',set())))

        def cabinet_dock(self,carrying):
            if carrying:return super().cabinet_dock(carrying)
            # Storage population varies across the shelf; the original bottle
            # demo's fixed x=2.10 dock leaves bowls in the far slot out of reach.
            target=self.bread_pose()[:3,3]
            rejected=[]
            # The open left-hinged panel occupies the straight-on base lane.
            # Try the opening side before more distant centered approaches.
            chosen = getattr(self, '_trial_storage_dock', None)
            docks = [chosen] if chosen is not None else cabinet_pickup_stances(self)
            for dock in docks:
                try:route=self.plan_route(dock[:2],False,face=float(dock[2]))
                except RuntimeError as exc:
                    rejected.append(str(exc));continue
                self._accepted_route=route
                self.task_navigate(dock[:2],False,face=float(dock[2]))
                self.record(cabinet_manipulation_dock=self.base_pose().tolist(),
                            cabinet_dock_target=target.tolist(),rejected_docks=rejected)
                return
            raise RuntimeError(f'No object-aligned cabinet pickup dock: {rejected}')

        def annotation_approach_allowed(self,pose):
            # Shelf retrieval needs side entry; do not apply breakfast's top-only filter.
            from cross_episode_sim.manipulation.grasp_qualification import GraspQualification
            from cross_episode_sim.fixtures.cabinet_transfer import front_shelf_approach
            return (GraspQualification.annotation_approach_allowed(self,pose)
                    and front_shelf_approach(pose,diagonal=True))

        def _plan_with_grasp_fallback(self):
            from cross_episode_sim.manipulation.recovery import RecoveringManipulation
            return RecoveringManipulation._plan_with_grasp_fallback(self)

    class DrawerBreakfast(DrawerTest, GatherBreakfast):
        drawer_cycle=DrawerLoop.drawer_cycle
        _drawer_cycle_once=DrawerLoop._drawer_cycle_once
        set_task_gripper_force=DrawerLoop.set_task_gripper_force
        object_label=OfficeBreakfastEpisode.object_label
        tick=GatherBreakfast.tick
        before_step=GatherBreakfast.before_step
        render_video_frame=GatherBreakfast.render_video_frame

        def pick_payload(self):
            from cross_episode_sim.skills.fixtures import StoragePickupSkill
            return StoragePickupSkill(self, drawer_pickup_stances(self)).run()

        deliver_payload=BreakfastEpisode.deliver_payload

        def surface_boxes(self,table):
            if table==DRAWER:
                gid=self.bottom_geom
                half=np.abs(self.data.geom_xmat[gid].reshape(3,3))@self.model.geom_size[gid]
                center=self.data.geom_xpos[gid]
                return [(gid,center-half,center+half)]
            from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
            return CrossRoomManipulation.surface_boxes(self,table)

        def prepare_pickup(self):
            self.tuck_for_navigation()
            approach_drawer_pickup(self)
            DrawerPickPlace.prepare_pickup(self)
            self.annotation_standoff=.12
            self.annotation_standoffs=(.12,.08)
            self.args.lift_height=.20
            # Breakfast's lower-rim recovery offsets are for vessels on open
            # tables. In a drawer they triple the library candidates with
            # contacts that drive the fingers toward its floor, and the bounded
            # planning budget runs out before the plain annotations.
            self.annotation_recovery_vertical_offsets=(0.,)

        def grasp_recovery_stances(self):
            current=self.base_pose()
            for dock in drawer_pickup_stances(self):
                if np.linalg.norm(dock-current)>.005:yield dock

        def plan_contact_path(self,stage,pose):
            self._drawer_departure_budget=drawer_departure_budget(self,stage,pose)
            try:
                path=super().plan_contact_path(stage,pose)
                if self._drawer_departure_budget:
                    self.record(drawer_departure_policy='separate pre-existing sub-0.1mm payload contact in first 2cm only')
                return path
            finally:self._drawer_departure_budget=None

        def navigation_penetration(self,data,carrying):
            return super().navigation_penetration(drawer_departure_contact_view(self,data,carrying),carrying)

        def interaction_label(self):
            return 'drawer handle' if self.operating_drawer else self.object_label()

        def gaze_target(self):
            return self.data.geom_xpos[self.handle_geom].copy() if self.operating_drawer else self.bread_pose()[:3,3]

        def stage_record(self,metrics):
            item=super().stage_record(metrics)
            if not self.operating_drawer:item['object']=self.object_name
            return item

    kind=info['source'];cls=CabinetBreakfast if kind=='cabinet' else DrawerBreakfast
    proxy=cls.__new__(cls);proxy.__dict__=controller.__dict__
    fixture=CabinetDoorTest if kind=='cabinet' else DrawerTest
    # The fixture adapters' initialization normally creates a fresh simulation;
    # bind only their fixture metadata onto the existing scene instead.
    saved={name:(name in controller.__dict__,controller.__dict__.get(name)) for name in
        ('door_joint','door_address','door_bids','handle_bids','handle_geom','articulating',
         'operating_door','operating_drawer','bottom_geom','drawer_force_limit',
         'dining','robot_actions',
         '_released_handle_state','_released_handle_base','_handle_grasp_cache',
         'annotation_adaptive_aperture','annotation_standoff','annotation_standoffs',
         'annotation_candidate_budget','initial_lift_height','active_family',
         'speculative_dock_trials')}
    saved_grasps=copy.deepcopy(getattr(controller,'successful_grasp_annotations',None))
    saved_args={name:(hasattr(controller.args,name),getattr(controller.args,name,None)) for name in
                ('motion_slowdown','grip_open','approach_policy','lift_height','annotation_source')}
    try:
        proxy.speculative_dock_trials=True
        proxy.door_joint=proxy.model.joint(fixture.fixture_joint).id
        proxy.door_address=proxy.model.jnt_qposadr[proxy.door_joint]
        proxy.door_bids=proxy.descendants(fixture.fixture_body)
        proxy.handle_bids=proxy.descendants(fixture.handle_body)
        proxy.handle_geom=proxy.model.geom(fixture.handle_geometry).id
        proxy.articulating=False;proxy.operating_door=False;proxy.operating_drawer=False
        # CabinetTransfer names its external destination "dining"; the gather
        # destination here is the kitchen filling counter.
        proxy.dining=proxy.task_supports['kitchen']
        # The existing action facade is bound to the original controller. Its
        # callbacks would bypass this adapter's storage-specific pickup. Use
        # the same underlying action methods directly for this scoped transfer.
        proxy.robot_actions=None
        proxy._handle_grasp_cache=[];proxy.__dict__.pop('_released_handle_state',None)
        if kind=='drawer':
            proxy.bottom_geom=proxy.model.geom('stack_1_main_group_4_inner_bottom').id
            proxy.drawer_force_limit=float(proxy.args.grip_force)
        cycle=proxy.door_cycle if kind=='cabinet' else proxy.drawer_cycle
        cycle(True)
        # Arm/world checks rebuild from the physically opened fixture.
        proxy.__dict__.pop('_base_map',None)
        BreakfastEpisode.transfer_role(proxy,info['role'])
        if not BreakfastEpisode.verify_role(proxy,info['role']):
            raise RuntimeError(f'Storage retrieval did not place {info["role"]} on the counter')
        cycle(False)
        proxy.__dict__.pop('_base_map',None)
    finally:
        # Shelf-library indices do not identify the original countertop poses.
        if kind=='cabinet':
            if saved_grasps is None:controller.__dict__.pop('successful_grasp_annotations',None)
            else:controller.successful_grasp_annotations=saved_grasps
        for name,(existed,value) in saved_args.items():
            if existed:setattr(controller.args,name,value)
            elif hasattr(controller.args,name):delattr(controller.args,name)
        for name,(existed,value) in saved.items():
            if existed:controller.__dict__[name]=value
            else:controller.__dict__.pop(name,None)
