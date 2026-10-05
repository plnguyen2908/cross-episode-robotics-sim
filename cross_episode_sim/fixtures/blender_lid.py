"""Native removable blender lid: remove/set down, regrasp/reseat, withdraw."""
import copy
import json
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.manipulation.cross_room import CrossRoomManipulation
from cross_episode_sim.paths import read_localized, FIXTURE_ASSETS_DIR

BLENDER = 'skill_blender_main'
LID = 'skill_blender_lid_main'
GENERATED = FIXTURE_ASSETS_DIR/'blender_lid.xml'


def cylinder_cover(radius,half):
    """Conservative sphere covers for a thin plate (37) or its handle (3)."""
    if half < radius*.1:
        spheres=[]
        for fraction,count in ((0.,1),(1/3,6),(2/3,12),(.95,18)):
            for angle in np.linspace(0.,2*np.pi,count,endpoint=False):
                spheres.append([radius*fraction*np.cos(angle),radius*fraction*np.sin(angle),
                                0.,np.hypot(.27*radius,half)])
        return np.asarray(spheres)
    return np.array([[0.,0.,z,np.hypot(radius,half/3)]
                     for z in (-2*half/3,0.,2*half/3)])


def lid_closed_state(position, rotation, anchor):
    """Mirror Blender.update_state + check_fxtr_upright (native 4 cm / 7 deg)."""
    angles = Rotation.from_matrix(rotation).as_euler('xyz', degrees=True)
    return bool(np.linalg.norm(np.asarray(position)-anchor) < .04
                and abs(angles[0]) < 7. and abs(angles[1]) < 7.)


def install_blender(scene, old_object, table, xyz):
    tree = ET.parse(scene); root = tree.getroot(); world = root.find('worldbody')
    native = ET.fromstring(read_localized(GENERATED))
    metadata = json.loads(GENERATED.with_suffix('.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bids = {model.body(table).id}
    for bid in range(model.nbody):
        if model.body_parentid[bid] in bids: bids.add(bid)
    tops = [float(data.geom_xpos[g,2]+model.geom_aabb[g,5])
            for g in range(model.ngeom) if model.geom_bodyid[g] in bids
            and (model.geom_contype[g] or model.geom_conaffinity[g])
            and .5 < data.geom_xpos[g,2] < 1.2]
    top = max(tops)
    position = np.array([xyz[0], xyz[1], top-metadata['blender_bottom_offset'][2]])
    blender = copy.deepcopy(native.find(f"worldbody/body[@name='{BLENDER}']"))
    lid = copy.deepcopy(native.find(f"worldbody/body[@name='{LID}']"))
    # Freeze unrelated controls as in the recorded scene export. The lid keeps
    # its authored free joint, damping, collision geometry and inertia.
    for body in blender.iter('body'):
        for joint in list(body.findall('joint')): body.remove(joint)
    blender.set('pos', ' '.join(map(str,position)))
    anchor = position+metadata['anchor_offset']
    lid.set('pos', ' '.join(map(str,anchor+[0.,0.,.015])))
    old = next(b for b in world if b.get('name') == old_object)
    world.remove(old); world.append(blender); world.append(lid)
    root.find('asset').extend(copy.deepcopy(list(native.find('asset'))))
    tree.write(scene)
    return LID, anchor+[0.,0.,.015]


def handle_annotations(output):
    native=ET.ElementTree(ET.fromstring(read_localized(GENERATED)))
    handle=native.find(".//geom[@name='skill_blender_lid_handle_main']")
    center=np.fromstring(handle.get('pos'),sep=' ')
    poses=[]
    for height in (0., .008):
        for yaw in np.linspace(0.,2*np.pi,8,endpoint=False):
            for tilt in (30.,15.,0.,45.):
                pose=np.eye(4)
                base=Rotation.from_euler('x',tilt,degrees=True).as_matrix()@np.array(
                    [[0.,-1.,0.],[0.,0.,-1.],[1.,0.,0.]])
                pose[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()@base
                pose[:3,3]=center+[0.,0.,height]
                poses.append(pose)
    path=output/'lid_handle_hypotheses.npz'
    np.savez_compressed(path,transforms=np.asarray(poses))
    return path


class BlenderLidTest(CrossRoomManipulation):
    video_filename='blender_lid_open_close.mp4'
    allow_supported_release_motion=True

    def __init__(self,args,selection):
        super().__init__(args,selection)
        self.counter=self.receptacles[0]
        self.source,self.destination=BLENDER,self.counter
        self.support_bids=self.table_bids[BLENDER]
        self.table_gids={g for g in range(self.model.ngeom)
                         if self.model.geom_bodyid[g] in self.support_bids
                         and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])}
        self.initial_lift_height=.025
        self.annotation_standoff=.07
        self.active_family='any'
        self.args.annotation_source='geometry_hypotheses'
        self.args.approach_policy='any-above-table'
        self.args.motion_slowdown=2.
        self.args.lift_height=.12
        self.asset_metadata=dict(asset='Blender008/Blender008_Lid',source='robocasa_fixture',
            model_xml=str(GENERATED.resolve()),object_scale=[1.,1.,1.])
        metadata=json.loads(GENERATED.with_suffix('.json').read_text())
        self.anchor=self.data.body(BLENDER).xpos.copy()+metadata['anchor_offset']
        self.dock=self.anchor[:2]+[0.,.44]
        self.report.update(task='native blender lid open and close',fixture_asset='Blender008 + Blender008_Lid; registry dimensions',
            tracked_objects=[LID],atomic_results=[],native_closed_position_tolerance_m=.04,
            native_closed_upright_tolerance_deg=7.,native_gripper_clearance_m=.15,
            initialization='native lid starts on blender; no task-time resets')

    def object_label(self): return 'blender lid'

    def annotation_approach_allowed(self,pose):
        if not super().annotation_approach_allowed(pose):
            return False
        # Keep the asymmetric Franka wrist above the worktop during a side
        # grasp, rather than using the jaw-equivalent wrist-below roll.
        return pose[2,0] > .5

    def bread_vertices(self):
        # A primitive's eight bounding-box corners do not sample its sides.
        # The pad-region grasp filter needs the actual cylindrical handle
        # surface, including points between its top and bottom caps.
        points=[]
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] not in self.bread_bids:
                continue
            if not (self.model.geom_contype[gid] or self.model.geom_conaffinity[gid]):
                continue
            if self.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CYLINDER:
                return super().bread_vertices()
            radius,half=self.model.geom_size[gid,:2]
            angles=np.linspace(0.,2*np.pi,64,endpoint=False)
            local=np.array([[radius*np.cos(a),radius*np.sin(a),z]
                            for z in np.linspace(-half,half,17) for a in angles])
            points.append(local@self.data.geom_xmat[gid].reshape(3,3).T+self.data.geom_xpos[gid])
        return np.concatenate(points) if points else super().bread_vertices()

    def gaze_target(self): return self.bread_pose()[:3,3]

    def update_recording_cameras(self):
        super().update_recording_cameras()
        self.cameras[0].lookat[:]=[*self.anchor[:2],1.0] if hasattr(self,'anchor') else self.bread_pose()[:3,3]
        self.cameras[0].distance=1.9; self.cameras[0].azimuth=110.; self.cameras[0].elevation=-20.
        self.cameras[1].lookat[:]=self.bread_pose()[:3,3]
        self.cameras[1].distance=.8;self.cameras[1].azimuth=100.;self.cameras[1].elevation=-20.

    def state(self):
        pose=self.bread_pose()
        return dict(lid_on_blender=lid_closed_state(pose[:3,3],pose[:3,:3],self.anchor),
            anchor_error_m=float(np.linalg.norm(pose[:3,3]-self.anchor)),
            gripper_distance_m=float(np.linalg.norm(self.tcp()[:3,3]-pose[:3,3])),
            upright_degrees=Rotation.from_matrix(pose[:3,:3]).as_euler('xyz',degrees=True).tolist())

    def prepare_pickup(self):
        self.tuck_for_navigation()
        dock=self.dock.copy()
        if self.source==self.counter:dock[0]=self.bread_pose()[0,3]
        self.task_navigate(dock,False,face=-np.pi/2)
        self.embodiment.open_gripper(self);self.tick(.5)
        self.untuck_for_manipulation()
        self.initial_lift_height=.025
        self.physically_rejected_annotation_variants=set()
        self.active_family='any'

    def transport_payload(self):
        # Short motion on one worktop: preserve live payload attachment, no base
        # carry or forced loaded tuck is needed between these neighboring targets.
        self.attach_native_lid()
        self.grasp_relative=np.linalg.inv(self.tcp())@self.bread_pose()
        self.destination_pose=self.closed_pose.copy()
        if self.destination==self.counter:
            self.destination_pose[0,3]-=.30
            local=(self.bread_vertices()-self.bread_pose()[:3,3])@self.bread_pose()[:3,:3]
            bottom=float((local@self.destination_pose[:3,:3].T)[:,2].min())
            boxes=self.surface_boxes(self.counter)
            xy=self.destination_pose[:2,3]
            top=max(high[2] for _,low,high in boxes if np.all(xy>low[:2]) and np.all(xy<high[:2]))
            self.destination_pose[2,3]=top-bottom
        self.support_bids=self.table_bids[self.destination]
        self.table_gids={g for g in range(self.model.ngeom)
                         if self.model.geom_bodyid[g] in self.support_bids
                         and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])}
        self.record(lid_destination=self.destination,object_target=self.destination_pose[:3,3].tolist())

    def attach_native_lid(self):
        # The shared whole-object box fills the empty space above the lid disk
        # beside the knob and falsely approaches the wrist. Cover the two
        # actual native cylinders separately, retaining conservative spheres.
        import torch
        from curobo.types import JointState,Pose
        obj=self.bread_pose();spheres=[]
        for gid in range(self.model.ngeom):
            if self.model.geom_bodyid[gid] not in self.bread_bids:
                continue
            if not (self.model.geom_contype[gid] or self.model.geom_conaffinity[gid]):continue
            if self.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CYLINDER:
                raise RuntimeError('Native lid attachment expects cylindrical collision primitives')
            cover=cylinder_cover(*self.model.geom_size[gid,:2])
            world=cover[:,:3]@self.data.geom_xmat[gid].reshape(3,3).T+self.data.geom_xpos[gid]
            cover[:,:3]=(world-obj[:3,3])@obj[:3,:3];spheres.extend(cover)
        if len(spheres)>40:raise RuntimeError('Lid cover exceeds planner attachment capacity')
        q=[float(self.data.joint('robot_0/'+n).qpos[0]) for n in self.planner.names]
        state=JointState.from_position(torch.tensor([q],device='cuda',dtype=torch.float32),joint_names=self.planner.names)
        pose=Pose(position=torch.tensor([obj[:3,3]-[0,0,self.embodiment.planner_tool_offset()]],device='cuda',dtype=torch.float32),
                  quaternion=torch.tensor([Rotation.from_matrix(obj[:3,:3]).as_quat(scalar_first=True)],device='cuda',dtype=torch.float32))
        before=self.planner.self_clearance(q)
        self.planner.detach_block()
        self.planner.attachments.update(torch.tensor(np.asarray(spheres),device='cuda',dtype=torch.float32),
            state,'attached_object_right',world_objects_pose_offset=pose)
        self.record(native_lid_cover_spheres=len(spheres),box_self_clearance_m=before,
                    native_cover_self_clearance_m=self.planner.self_clearance(q))

    def place_payload(self):
        self.review_phase='SET LID ON WORKTOP' if self.destination==self.counter else 'RESEAT LID ON BLENDER'
        # Validate the full descent on a separate data instance before moving
        # the live arm. A reachable overhead pose alone is not sufficient.
        offsets=([(.0,-.18),(.0,-.10),(.0,.0),(.10,-.18),(-.10,-.18)]
                 if self.destination==self.counter else [(0.,0.)])
        live=self.data
        addresses=[self.model.jnt_qposadr[self.model.joint('robot_0/'+n).id]
                   for n in self.planner.names]
        rejected=[]
        for dx,dy in offsets:
            object_pose=self.destination_pose.copy();object_pose[:2,3]+=[dx,dy]
            target=object_pose@np.linalg.inv(self.grasp_relative)
            # A round removable lid has no required yaw. Face the held grasp
            # toward the destination instead of preserving an awkward pickup
            # azimuth that can fold the elbow against the worktop.
            delta=object_pose[:2,3]-self.base_pose()[:2]
            yaw=np.arctan2(delta[1],delta[0])-np.arctan2(target[1,2],target[0,2])
            object_pose[:3,:3]=Rotation.from_euler('z',yaw).as_matrix()@object_pose[:3,:3]
            if self.destination==self.counter:
                target=object_pose@np.linalg.inv(self.grasp_relative)
                # Set the disk down on its edge first. This permits the wrist
                # to stay above the counter; the released free lid settles
                # flat under gravity rather than forcing a horizontal wrist.
                tilt=Rotation.from_rotvec(-np.radians(25.)*target[:3,1]).as_matrix()
                object_pose[:3,:3]=tilt@object_pose[:3,:3]
                local=(self.bread_vertices()-self.bread_pose()[:3,3])@self.bread_pose()[:3,:3]
                bottom=float((local@object_pose[:3,:3].T)[:,2].min())
                xy=object_pose[:2,3]
                top=max(high[2] for _,low,high in self.surface_boxes(self.counter)
                        if np.all(xy>low[:2]) and np.all(xy<high[:2]))
                object_pose[2,3]=top-bottom
            elif hasattr(self,'align_lid_center'):
                object_pose=self.align_lid_center(object_pose)
            target=object_pose@np.linalg.inv(self.grasp_relative)
            target[2,3]+=.02 if self.destination==self.counter else getattr(self,'lid_release_clearance',.004)
            above=target.copy();above[2,3]+=.08 if self.destination==self.counter else .04
            try:
                goal=list(above[:3,3]-[0,0,self.embodiment.planner_tool_offset()])+list(
                    Rotation.from_matrix(above[:3,:3]).as_quat(scalar_first=True))
                trajectory=self.planner.plan(live.qpos[addresses].tolist(),goal)
                probe=mujoco.MjData(self.model);probe.qpos[:]=live.qpos
                probe.time=live.time
                probe.qpos[addresses]=trajectory[-1]
                payload=above@self.grasp_relative
                probe.joint(self.object_joint).qpos[:3]=payload[:3,3]
                probe.joint(self.object_joint).qpos[3:]=Rotation.from_matrix(payload[:3,:3]).as_quat(scalar_first=True)
                mujoco.mj_forward(self.model,probe)
                self.data=probe
                self.plan_contact_path('preflight lid seating',target)
                self.preplanned_moves['above lid destination']=(above,trajectory)
                break
            except RuntimeError as exc:
                rejected.append(str(exc))
            finally:
                self.data=live
        else:
            raise RuntimeError(f'No complete lid placement trajectory: {rejected}')
        self.record(lid_placement_preflight=True,rejected_placements=rejected,
                    selected_object_target=object_pose[:3,3].tolist())
        self.move('above lid destination',above)
        self.mesh_contact_move('seat blender lid',target)
        self.holding_loaf=False
        self.embodiment.open_gripper(self);self.tick(1.)
        self.planner.detach_block();self.attached=False
        support=self.assignment()[self.object_name]
        if support!=self.destination:raise RuntimeError(f'Lid release support {support} != {self.destination}')
        self.record(lid_placement_support=support)
        start=self.tcp().copy()
        horizontal=-start[:3,2].copy();horizontal[2]=0.
        horizontal/=max(np.linalg.norm(horizontal),1e-8)
        errors=[]
        for offset in (-.12*start[:3,2],-.06*start[:3,2],
                       .12*horizontal+[0.,0.,.02],.08*horizontal):
            retreat=start.copy();retreat[:3,3]+=offset
            try:
                path=self.plan_contact_path('preflight lid withdrawal',retreat)
                break
            except RuntimeError as exc:errors.append(str(exc))
        else:raise RuntimeError(f'No clear released-lid withdrawal: {errors}')
        self.record(lid_withdrawal_offset=np.asarray(offset).tolist(),rejected_withdrawals=errors)
        self.mesh_contact_move('withdraw from released blender lid',retreat,path=path)
        self.tuck_for_navigation();self.tick(.5)
        state=self.state()
        valid=(state['gripper_distance_m']>.15 and self.assignment()[self.object_name]==self.destination)
        valid=valid and (not state['lid_on_blender'] if self.destination==self.counter else state['lid_on_blender'])
        if not valid:raise RuntimeError(f'Native lid task state did not pass after release: {state}')
        self.report['atomic_results'].append(dict(skill='open_lid' if self.destination==self.counter else 'close_lid',success=True,**state))
        self.report['success']=True
        self.record(atomic_lid_success=True,**state)

    def run_test(self):
        success=False
        try:
            self.tick(1.)
            self.closed_pose=self.bread_pose().copy()
            if not self.state()['lid_on_blender'] or self.assignment()[self.object_name]!=BLENDER:
                raise RuntimeError(f'Lid must start physically seated: {self.state()}')
            self.report['initial_state']=self.state()
            self.tuck_for_navigation()
            for source,destination in ((BLENDER,self.counter),(self.counter,BLENDER)):
                self.source,self.destination=source,destination
                self.support_bids=self.table_bids[source]
                self.table_gids={g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] in self.support_bids
                                 and (self.model.geom_contype[g] or self.model.geom_conaffinity[g])}
                self.initial_object_pose=self.bread_pose().copy()
                self.transfer_start=self.data.joint(self.object_joint).qpos.copy()
                self.preplanned_moves={};self._recovery_exhausted=False
                self.review_phase='OPEN BLENDER LID' if source==BLENDER else 'CLOSE BLENDER LID'
                self.execute_transfer()
            success=len(self.report['atomic_results'])==2 and self.in_default_travel_posture(False)
        except Exception as exc:
            self.report.update(error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
        finally:
            self.report.update(success=success,final_state=self.state(),scope='one native removable blender lid, open/set down and close/reseat; no blending')
            self.finish_run_outputs(success)
        return 0 if success else 1
