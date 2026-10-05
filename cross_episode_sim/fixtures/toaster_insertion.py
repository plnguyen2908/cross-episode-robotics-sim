"""Native bread pickup and constrained insertion into a native toaster slot."""
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

TOASTER = 'skill_toaster_main'
GENERATED = FIXTURE_ASSETS_DIR/'toaster.xml'


def collision_vertices(model, data, bodies):
    points = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in bodies or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]; a = model.mesh_vertadr[mesh]; n = model.mesh_vertnum[mesh]
            local = model.mesh_vert[a:a+n]
        else:
            corners = np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
            local = model.geom_aabb[g,:3]+corners*model.geom_aabb[g,3:]
        points.extend(local@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])
    return np.asarray(points)


def install_toaster(scene, obj, table, xyz):
    tree = ET.parse(scene); root = tree.getroot(); world = root.find('worldbody')
    native = ET.ElementTree(ET.fromstring(read_localized(GENERATED)))
    metadata = json.loads(GENERATED.with_suffix('.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model); mujoco.mj_forward(model,data)
    # Same physical surface filter as CrossRoomManipulation.surface_boxes:
    # exported fixtures also contain hidden registration geometry at z=10 m.
    tops=[]
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    for g in range(model.ngeom):
        if model.geom_bodyid[g]!=model.body(table).id or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
        vertices=(model.geom_aabb[g,:3]+corners*model.geom_aabb[g,3:])@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        low,high=vertices.min(0),vertices.max(0)
        if .3<high[2]<1.3 and high[2]-low[2]<.15:tops.append(high[2])
    if not tops:raise ValueError('No physical tabletop for toaster')
    top=max(tops)
    body = copy.deepcopy(native.find('worldbody/body'))
    for part in body.iter('body'):
        for joint in list(part.findall('joint')):part.remove(joint)
    position = np.array([xyz[0]-.12,xyz[1]-.05,top-metadata['bottom_offset'][2]])
    body.set('pos',' '.join(map(str,position)));world.append(body)
    root.find('asset').extend(copy.deepcopy(list(native.find('asset'))))
    # Authored initial pose: a slice standing on its edge, as in toaster tasks.
    # It remains a free body and must settle, lift and hold under real contact.
    bread = world.find(f"body[@name='{obj}']")
    verts = collision_vertices(model,data,{model.body(obj).id})
    old = data.body(obj);local = (verts-old.xpos)@old.xmat.reshape(3,3)
    R = Rotation.from_euler('y',90,degrees=True).as_matrix()
    rotated = local@R.T
    bread_pos = np.array([xyz[0]+.14,xyz[1],top-rotated[:,2].min()+.001])
    bread.set('pos',' '.join(map(str,bread_pos)))
    bread.set('quat',' '.join(map(str,Rotation.from_matrix(R).as_quat(scalar_first=True))))
    tree.write(scene)
    return bread_pos


def edge_annotations(scene, obj, output):
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    vertices=collision_vertices(model,data,{model.body(obj).id})
    low,high=vertices.min(0),vertices.max(0)
    object_pose=np.eye(4);object_pose[:3,:3]=data.body(obj).xmat.reshape(3,3);object_pose[:3,3]=data.body(obj).xpos
    poses=[]
    for tilt in (-45.,-60.,-30.):
        R=Rotation.from_euler('x',tilt,degrees=True).as_matrix()@np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
        for depth in (.006,.012,.018):
            for dy in (0.,.012,-.012):
                pose=np.eye(4);pose[:3,:3]=R
                center=(low+high)/2;center[1]+=dy;center[2]=high[2]-depth
                pose[:3,3]=center+.019*R[:,2]
                poses.append(np.linalg.inv(object_pose)@pose)
    path=Path(output)/'bread_edge_hypotheses.npz'
    np.savez_compressed(path,transforms=poses)
    return path


class ToasterInsertionTest(CrossRoomManipulation):
    video_filename='toaster_insertion.mp4'

    def __init__(self,args,selection):
        super().__init__(args,selection)
        self.source,self.destination=self.receptacles
        self.support_bids=self.table_bids[self.source]
        self.slot_floor=self.model.geom('skill_toaster_slotR_floor').id
        self.slot_region=self.model.geom('skill_toaster_reg_slotR').id
        self.report.update(task='pick native bread slice and insert into toaster slot',
            atomic_results=[],fixture_asset='Toaster033',object_initial_pose='upright on worktop, free body',
            success_predicate='native slot floor contact, footprint inside slot, insertion depth, gripper released and withdrawn')
        self.active_family='any'

    def prepare_pickup(self):
        self.tuck_for_navigation()
        dock=self.data.body(TOASTER).xpos[:2]+[.10,.45]
        self.task_navigate(dock,False,face=-np.pi/2)
        self.embodiment.open_gripper(self);self.tick(.5)
        self.untuck_for_manipulation()
        self.active_family='any'

    def transport_payload(self):
        self.review_phase='ALIGN BREAD WITH TOASTER SLOT'
        self.prepare_loaded_manipulation()
        self.grasp_relative=np.linalg.inv(self.tcp())@self.bread_pose()
        self.support_bids=self.table_bids[TOASTER]

    def insertion_state(self):
        vertices=self.bread_vertices()
        frame=self.data.geom_xmat[self.slot_region].reshape(3,3)
        relative=(vertices-self.data.geom_xpos[self.slot_region])@frame
        half=self.model.geom_size[self.slot_region]
        inside=bool(np.all(np.abs(relative[:,:2])<=half[:2]+.001))
        floor_contact=any(self.slot_floor in (c.geom1,c.geom2) and
            self.model.geom_bodyid[c.geom2 if c.geom1==self.slot_floor else c.geom1] in self.bread_bids
            for c in self.data.contact)
        depth=self.rim_height-float(vertices[:,2].min())
        return dict(slot_floor_contact=floor_contact,footprint_inside=inside,insertion_depth_m=depth,
                    gripper_distance_m=float(np.linalg.norm(self.tcp()[:3,3]-self.bread_pose()[:3,3])))

    def insertion_rotation(self):
        return Rotation.from_euler('y',90,degrees=True).as_matrix()

    def reach_insertion_approach(self,obj,local,inv):
        rotated=local@obj[:3,:3].T
        above=obj.copy();above[2,3]+=max(.10,self.rim_height+.035-(rotated[:,2].min()+obj[2,3]))
        self.move('align bread above toaster slot',above@inv)
        return None

    def place_payload(self):
        self.review_phase='INSERT BREAD INTO NATIVE SLOT'
        held=self.bread_pose();local=(self.bread_vertices()-held[:3,3])@held[:3,:3]
        obj=held.copy();obj[:3,:3]=self.insertion_rotation()
        rotated=local@obj[:3,:3].T
        floor=self.data.geom_xpos[self.slot_floor].copy()
        obj[:2,3]=floor[:2]-(rotated.min(0)+rotated.max(0))[:2]/2
        floor_top=floor[2]+self.model.geom_size[self.slot_floor,2]
        obj[2,3]=floor_top-rotated[:,2].min()+.015
        self.rim_height=self.data.body(TOASTER).xpos[2]+.095
        inv=np.linalg.inv(self.grasp_relative)
        path=self.reach_insertion_approach(obj,local,inv)
        self.mesh_contact_move('lower bread into toaster slot',obj@inv,path=path)
        before=self.insertion_state();self.record(insertion_before_release=before)
        if not before['footprint_inside'] or before['insertion_depth_m']<.04:
            raise RuntimeError('Bread did not enter the slot before release')
        self.holding_loaf=False;self.embodiment.open_gripper(self);self.tick(1.)
        self.planner.detach_block();self.attached=False
        state=self.insertion_state();self.record(insertion_after_release=state)
        if not state['slot_floor_contact'] or not state['footprint_inside']:
            raise RuntimeError('Released bread is not seated on the native slot floor')
        self.withdraw_after_insertion()
        self.report['atomic_results'].append(dict(skill='insert_bread_into_slot',success=True,**self.insertion_state()))

    def withdraw_after_insertion(self):
        retreat=self.tcp().copy();retreat[2,3]+=.10
        self.mesh_contact_move('withdraw above inserted bread',retreat)
        self.tuck_for_navigation();self.tick(.5)

    def run_test(self):
        success=False
        try:
            self.tick(1.)
            self.initial_object_pose=self.bread_pose().copy()
            self.transfer_start=self.data.joint(self.object_joint).qpos.copy()
            self.review_phase='PICK BREAD FROM WORKTOP'
            self.execute_transfer()
            state=self.insertion_state()
            success=bool(state['slot_floor_contact'] and state['footprint_inside'] and state['insertion_depth_m']>.04
                         and state['gripper_distance_m']>.15 and self.in_default_travel_posture(False))
            self.report['final_insertion_state']=state
        except Exception as exc:
            self.report.update(error=str(exc),traceback=traceback.format_exc());traceback.print_exc()
        finally:
            self.report.update(success=success,scope='one native slice and slot; physical insertion, no toaster heating or retrieval')
            self.finish_run_outputs(success)
        return 0 if success else 1
