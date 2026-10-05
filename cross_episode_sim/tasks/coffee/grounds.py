"""Coarse granular coffee and a retrofit hopper; no fluid/thermal simulation."""
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

HOPPER = 'coffee_grounds_hopper'
PARTICLE_RADIUS = .0035
PARTICLE_MASS = .00010


def make_dosing_box(body):
    """Replace a vessel with an 80 x 70 x 75 mm open tin and a pinch handle."""
    for child in list(body):
        if child.tag not in ('joint', 'freejoint'):
            body.remove(child)
    body.set('quat', '1 0 0 0')
    # Four separate walls and a floor leave a real open cavity. The short
    # external tab gives the fingers purchase without entering the grounds.
    pieces = [([0,0,-.0355],[.04,.035,.002]),
              ([-.038,0,0],[.002,.035,.0375]),
              ([.038,0,0],[.002,.035,.0375]),
              ([0,-.033,0],[.036,.002,.0375]),
              ([0,.033,0],[.036,.002,.0375]),
              ([.059,0,.019],[.021,.01,.006])]
    for i,(pos,size) in enumerate(pieces):
        attrs=dict(type='box',pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),
                   rgba='.62 .32 .12 1',friction='.95 .01 .001',solref='.004 1')
        ET.SubElement(body,'geom',name=f'coffee_dosing_box_collision_{i}',group='0',
                      mass=str(.10/len(pieces)),**attrs)
        ET.SubElement(body,'geom',name=f'coffee_dosing_box_visual_{i}',group='1',
                      mass='0',contype='0',conaffinity='0',**attrs)
    return dict(shape='box',center_xy=[0.,0.],half_width_xy=[.036,.031],
                radius=.031,bottom_z=-.0335,rim_z=.0375,
                reference_rotation=np.eye(3).tolist())


def box_grasp_annotations(path):
    """Geometry hypotheses for the external tab; never registry-qualified."""
    poses=[]
    side=np.array([[0.,0.,-1.],[0.,1.,0.],[1.,0.,0.]])
    for tilt in (15.,10.,20.,25.):
        for x in (.068,.074,.062):
            pose=np.eye(4)
            pose[:3,:3]=Rotation.from_euler('y',-tilt,degrees=True).as_matrix()@side
            pose[:3,3]=[x,0.,.019]
            poses.append(pose)
    np.savez_compressed(path,transforms=np.asarray(poses))


def add_hopper(world, center, bottom, top):
    """An open physical receiving bin, not a region inside a solid mesh."""
    body = ET.SubElement(world, 'body', name=HOPPER, pos=f'{center[0]} {center[1]} {bottom}')
    half = .054
    height = top-bottom
    specs = [([0,0,.004],[half+.008,half+.008,.004]),
             ([-half-.004,0,height/2],[.004,half+.008,height/2]),
             ([half+.004,0,height/2],[.004,half+.008,height/2]),
             ([0,-half-.004,height/2],[half,.004,height/2]),
             ([0,half+.004,height/2],[half,.004,height/2])]
    for i,(pos,size) in enumerate(specs):
        attrs=dict(type='box',pos=' '.join(map(str,pos)),size=' '.join(map(str,size)),
                   rgba='.24 .25 .27 1',friction='.6 .005 .0001',solref='.004 1')
        ET.SubElement(body,'geom',name=f'{HOPPER}_collision_{i}',group='0',**attrs)
        ET.SubElement(body,'geom',name=f'{HOPPER}_visual_{i}',group='1',
                      contype='0',conaffinity='0',mass='0',**attrs)
    return dict(body=HOPPER,center_xy=list(center),bottom_z=bottom+.008,top_z=top,half_width=half)


def grain_offsets(cavity, count=32):
    if type(count) is not int or not 8 <= count <= 48:
        raise ValueError('Coffee test supports 8–48 coarse granules')
    r = PARTICLE_RADIUS
    points = []
    for layer in range(3):
        for x in (-1.5,-.5,.5,1.5):
            for y in (-1.5,-.5,.5,1.5):
                xy = np.array([x,y])*(2*r+.002)
                if np.linalg.norm(xy)+r+.002 > cavity['radius']:
                    raise ValueError('Dosing vessel too narrow for the initial granule packing')
                z = cavity['bottom_z']+r+.003+layer*(2*r+.002)
                if z+r >= cavity['rim_z']-.008:
                    raise ValueError('Dosing vessel too shallow for grounds')
                points.append([*(np.asarray(cavity['center_xy'])+xy),z])
                if len(points)==count:return np.asarray(points)


def add_grounds(world, vessel_position, cavity, count=32, vessel_rotation=None):
    names = []
    rotation=np.eye(3) if vessel_rotation is None else np.asarray(vessel_rotation)
    for i,offset in enumerate(grain_offsets(cavity,count)):
        name=f'coffee_grain_{i:03d}';names.append(name)
        body=ET.SubElement(world,'body',name=name,pos=' '.join(map(str,np.asarray(vessel_position)+rotation@offset)))
        ET.SubElement(body,'freejoint',name=name+'_joint')
        attrs=dict(type='sphere',size=str(PARTICLE_RADIUS),rgba='.16 .07 .025 1')
        ET.SubElement(body,'geom',name=name+'_collision',group='0',mass=str(PARTICLE_MASS),
                      friction='.45 .002 .0001',solref='.004 1',**attrs)
        ET.SubElement(body,'geom',name=name+'_visual',group='1',mass='0',contype='0',conaffinity='0',**attrs)
    return names


def grain_inventory(model, data, spec, grain_ids=None):
    """Disjoint measured sets; spilled particles remain in the scene."""
    hopper=spec['hopper'];vessel=data.body(spec['dosing_body'])
    cavity=spec['dosing_cavity'];reference=np.asarray(cavity['reference_rotation'])
    if grain_ids is None:
        grain_ids=np.array([model.body(name).id for name in spec['grains']],dtype=int)
    p=data.xpos[grain_ids]
    local=(p-vessel.xpos)@vessel.xmat.reshape(3,3)@reference.T
    in_hopper=(np.all(np.abs(p[:,:2]-hopper['center_xy']) <= hopper['half_width']-PARTICLE_RADIUS+.001,axis=1)
               & (hopper['bottom_z']-.001 <= p[:,2]-PARTICLE_RADIUS)
               & (p[:,2]+PARTICLE_RADIUS <= hopper['top_z']+.001))
    if cavity.get('shape')=='box':
        inside_xy=np.all(np.abs(local[:,:2]-cavity['center_xy']) <=
                         np.asarray(cavity['half_width_xy'])+PARTICLE_RADIUS,axis=1)
    else:
        inside_xy=np.linalg.norm(local[:,:2]-cavity['center_xy'],axis=1) < cavity['radius']+PARTICLE_RADIUS
    in_vessel=(inside_xy
               & (cavity['bottom_z']-.005 <= local[:,2])
               & (local[:,2] <= cavity['rim_z']+PARTICLE_RADIUS))
    result=dict(hopper=[],vessel=[],spilled=[])
    for name,captured,contained in zip(spec['grains'],in_hopper,in_vessel):
        result['hopper' if captured else 'vessel' if contained else 'spilled'].append(name)
    return result


def pour_pose(spec, angle_deg, yaw=0.):
    """Keep the pouring lip above the opening while tipping toward its center."""
    cavity=spec['dosing_cavity'];hopper=spec['hopper']
    pivot=np.array([*cavity['center_xy'],cavity['rim_z']])
    pivot[1]+=cavity['half_width_xy'][1] if cavity.get('shape')=='box' else cavity['radius']
    R=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@Rotation.from_euler('x',-angle_deg,degrees=True).as_matrix()
    pose=np.eye(4)
    pose[:3,:3]=R@np.asarray(cavity['reference_rotation'])
    # Clearance for the complete upright vessel before tipping, not only its
    # rim. A low lip target puts the vessel bottom through the hopper wall.
    upright_height=max(.12,cavity['rim_z']-cavity['bottom_z']+.025)
    # Once tilted, the cup bottom swings away from the inlet. Lower the lip
    # progressively to avoid accelerating granules across a long free fall.
    clearance=spec.get('pour_lip_clearance_m',.035)
    height=upright_height-(upright_height-clearance)*np.clip((angle_deg-35.)/40.,0.,1.)
    lip=np.array([*hopper['center_xy'],hopper['top_z']+height])
    pose[:3,3]=lip-R@pivot
    return pose


@dataclass
class BrewState:
    """Task-state surrogate: contact-triggered brewing with explicit prerequisites."""
    duration: float = 5.
    started_at: float | None = None
    completed: bool = False
    pressed_before: bool = False
    aborted: bool = False

    def update(self, time, pressed, grounds_ready, lid_closed, mug_ready):
        ready=grounds_ready and lid_closed and mug_ready
        event=None
        if pressed and not self.pressed_before and self.started_at is None and not self.completed:
            if ready:
                self.started_at=time;self.aborted=False;event='brew_started'
            else:event='start_rejected_missing_prerequisite'
        if self.started_at is not None and not self.completed:
            if not ready:
                self.started_at=None;self.aborted=True;event='brew_aborted'
            elif not pressed and time-self.started_at>=self.duration:
                self.completed=True;event='brew_complete'
        self.pressed_before=bool(pressed)
        return event
