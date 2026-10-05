"""Register the existing FR3 arm and measured TidyBot++ base in RoboSuite.

Base dynamics use a planar, velocity-actuated abstraction (not wheel dynamics).
The hand reuses the existing MolmoSpaces Robotiq 2f85 v4 model.
"""
from pathlib import Path
import os
import copy
import xml.etree.ElementTree as ET
import numpy as np
from robosuite.models.bases import MobileBaseModel, register_base
from robosuite.models.robots.manipulators.panda_robot import Panda
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.robots import register_robot_class
from robosuite.models.grippers import GripperModel,register_gripper
from cross_episode_sim.paths import ASSETS_DIR, GENERATED_DIR

TUCK=np.array([0.,0.,0.,-.20,0.,.60,0.])


def prepare_models():
    generated=GENERATED_DIR;generated.mkdir(parents=True,exist_ok=True)
    configured=os.environ.get('MLSPACES_ASSETS_DIR')
    candidates=([Path(configured)/'robots/franka_droid/model.xml'] if configured else [])
    candidates+=list((Path.home()/'.cache/molmospaces/assets').glob('*/robots/franka_droid/model.xml'))
    source=next((p for p in candidates if p.is_file()),None)
    if source is None:raise FileNotFoundError('Set MLSPACES_ASSETS_DIR to the existing FR3 asset installation')
    arm=ET.parse(source).getroot()
    arm.find('./worldbody/body').set('name','base')
    for mesh in arm.findall('./asset/mesh'):
        mesh.set('name',mesh.get('name',Path(mesh.get('file')).stem))
        mesh.set('file',str((source.parent/'assets'/mesh.get('file')).resolve()))
    for parent in arm.iter():
        for node in list(parent):
            if node.tag in ('attach','frame','model','keyframe'):parent.remove(node)
    ET.SubElement(arm.find(".//body[@name='fr3_link0']"),'site',name='right_center',size='.001',rgba='0 0 0 0')
    link7=arm.find(".//body[@name='fr3_link7']")
    ET.SubElement(link7,'body',name='right_hand',pos='0 0 .107',quat='.7071068 0 0 .7071068')
    for node in arm.findall('.//geom'):
        node.set('group','1' if node.get('class')=='visual' else '0')
    for node in arm.iter():
        node.attrib.pop('childclass',None)
        cls=node.attrib.pop('class',None)
        if node.tag=='geom':
            node.set('type','mesh')
            if cls=='visual':node.set('contype','0');node.set('conaffinity','0')
            else:node.set('mass','0')
        if node.tag=='joint':node.set('armature','.1');node.set('damping','1')
    arm.remove(arm.find('default'))
    actuator=arm.find('actuator');actuator.clear()
    for i in range(1,8):
        limit=87 if i<5 else 12
        ET.SubElement(actuator,'motor',name=f'torque{i}',joint=f'fr3_joint{i}',ctrllimited='true',ctrlrange=f'-{limit} {limit}')
    ET.ElementTree(arm).write(generated/'fr3.xml')
    src=ASSETS_DIR/'tidybot_base/models/stanford_tidybot/base.xml'
    base=ET.parse(src).getroot()
    for mesh in base.findall('./asset/mesh'):
        mesh.set('name',Path(mesh.get('file')).stem)
        mesh.set('file',str((src.parent/'../assets'/mesh.get('file')).resolve()))
    body=base.find('./worldbody/body');body.set('name','base')
    ET.SubElement(body,'site',name='center',size='.001',rgba='0 0 0 0')
    ET.SubElement(body,'camera',name='frontview',pos='.21 0 .30',xyaxes='0 -1 0 0 0 1',fovy='80')
    body.remove(body.find('inertial'))
    for j in list(body.findall('joint')):body.remove(j)
    for g in body.findall('geom'):
        if g.get('class')=='collision':body.remove(g)
        else:g.set('group','1')
    ET.SubElement(body,'geom',name='base_collision',type='box',size='.2405 .2405 .154',pos='0 0 .181',group='0',mass='60')
    ET.SubElement(body,'body',name='support',pos='0 0 .335')
    actuator=base.find('actuator');actuator.clear()
    for suffix,kind,axis in [('forward','slide','1 0 0'),('side','slide','0 1 0'),('yaw','hinge','0 0 1')]:
        name='joint_mobile_'+suffix
        ET.SubElement(body,'joint',name=name,type=kind,axis=axis,limited='false',damping='1')
        ET.SubElement(actuator,'velocity',name='actuator_mobile_'+suffix,joint=name,kv='1000',ctrllimited='true',ctrlrange='-.5 .5',forcelimited='true',forcerange='-300 300')
    ET.ElementTree(base).write(generated/'tidybot.xml')
    hand_source=source.parent/'robotiq_2f85_v4/2f85.xml'
    hand=ET.parse(hand_source).getroot()
    defaults={}
    def gather(node,parent):
        values=copy.deepcopy(parent)
        for child in node:
            if child.tag!='default':values.setdefault(child.tag,{}).update(child.attrib)
        defaults[node.get('class','main')]=values
        for child in node.findall('default'):gather(child,values)
    gather(hand.find('default'),{})
    def flatten(node,inherited='main'):
        cls=node.get('class',inherited)
        attrs=defaults.get(cls,{}).get(node.tag,{})
        for key,value in attrs.items():
            if key not in node.attrib:node.set(key,value)
        next_class=node.get('childclass',inherited)
        node.attrib.pop('class',None);node.attrib.pop('childclass',None)
        for child in node:
            if child.tag!='default':flatten(child,next_class)
    flatten(hand);hand.remove(hand.find('default'))
    for mesh in hand.findall('./asset/mesh'):
        mesh.set('name',Path(mesh.get('file')).stem)
        mesh.set('file',str((hand_source.parent/'assets'/mesh.get('file')).resolve()))
    for geom in hand.findall('.//geom'):
        geom.set('group','1' if geom.get('contype')=='0' else '0')
    root=hand.find('./worldbody/body')
    eef=ET.SubElement(root,'body',name='eef',pos='0 0 .155',quat='.7071068 0 0 -.7071068')
    for name in ('grip_site','grip_site_cylinder','ee','ee_x','ee_y','ee_z'):
        ET.SubElement(eef,'site',name=name,size='.001',rgba='0 0 0 0')
    ET.SubElement(root,'site',name='ft_frame',size='.001',rgba='0 0 0 0')
    sensor=ET.SubElement(hand,'sensor')
    ET.SubElement(sensor,'force',name='force_ee',site='ft_frame')
    ET.SubElement(sensor,'torque',name='torque_ee',site='ft_frame')
    ET.ElementTree(hand).write(generated/'robotiq.xml')
    return generated


@register_base
class TidyBotBase(MobileBaseModel):
    def __init__(self,idn=0):super().__init__(str(prepare_models()/'tidybot.xml'),idn=idn)
    @property
    def top_offset(self):return np.zeros(3)
    @property
    def horizontal_radius(self):return np.hypot(.2405,.2405)


@register_robot_class('WheeledRobot')
class TidyBotFranka(Panda):
    def __init__(self,idn=0):
        ManipulatorModel.__init__(self,str(prepare_models()/'fr3.xml'),idn=idn)
    @property
    def default_base(self):return 'TidyBotBase'
    @property
    def default_gripper(self):return {'right':'TidyBotRobotiq85'}
    @property
    def default_arms(self):return {'right':'Panda'}
    @property
    def init_qpos(self):return TUCK.copy()
    @property
    def base_xpos_offset(self):return {'empty':(-.6,0,0),'table':lambda length:(-.16-length/2,0,0)}


def controller_config():
    import json,robosuite
    path=Path(robosuite.__file__).parent/'controllers/config/robots/default_pandaomron.json'
    cfg=json.loads(path.read_text());cfg['body_parts'].pop('torso')
    cfg['body_parts']['arms']['right'].update(type='JOINT_POSITION',input_type='absolute',input_min=-4.6,input_max=4.6,output_min=-4.6,output_max=4.6,kp=180,damping_ratio=1)
    cfg['composite_controller_specific_configs']={'body_part_ordering':['right','right_gripper','base']}
    cfg['body_parts'].update(cfg['body_parts'].pop('arms'))
    return cfg


@register_gripper
class TidyBotRobotiq85(GripperModel):
    def __init__(self,idn=0):super().__init__(str(prepare_models()/'robotiq.xml'),idn=idn)
    def format_action(self,action):return np.clip(np.asarray(action),-1,1)
    @property
    def init_qpos(self):return np.zeros(len(self.joints))
    @property
    def _important_geoms(self):
        return {key:[f'{side}_pad1',f'{side}_pad2'] for side in ('left','right') for key in (side+'_finger',side+'_fingerpad')}
