"""Shared initial-placement policy for accessible rectangular tabletop edges."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.manipulation.edge_access import tabletop_bounds, object_bodies, collision_vertices


def supported_edge_position(vertices, table_vertices, inset=.02):
    """Place the footprint 2 cm inside the +Y edge, centered across its width."""
    lo,hi=table_vertices.min(0),table_vertices.max(0)
    low,high=vertices.min(0),vertices.max(0)
    if np.any(high[:2]-low[:2] > hi[:2]-lo[:2]-2*inset):
        raise ValueError('Object footprint does not fit the supported edge site')
    return np.array([(lo[0]+hi[0]-low[0]-high[0])/2,
                     hi[1]-inset-high[1],hi[2]+.001-low[2]])


def install_edge_scene(scene,obj,table):
    """Author only the initial pose; native mass, shape and joints stay intact."""
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    vertices=collision_vertices(m,d,object_bodies(m,obj))
    surface=tabletop_bounds(m,d,object_bodies(m,table))
    position=d.body(obj).xpos.copy();R=d.body(obj).xmat.reshape(3,3).copy()
    # Use the same heading across assets while preserving their authored upright/flat axis.
    delta=Rotation.from_euler('z',np.radians(12)-np.arctan2(R[1,0],R[0,0])).as_matrix()
    rotated=(vertices-position)@delta.T
    xyz=supported_edge_position(rotated,surface)
    tree=ET.parse(scene);body=tree.find(f".//body[@name='{obj}']")
    body.set('pos',' '.join(map(str,xyz)))
    body.set('quat',' '.join(map(str,Rotation.from_matrix(delta@R).as_quat(scalar_first=True))))
    tree.write(scene)
    (Path(scene).parent/'edge_placement.json').write_text(json.dumps(dict(
        policy='supported_front_edge_v1',inset_m=.02,yaw_degrees=12,
        object=obj,support=table,position=xyz.tolist(),initial_pose_only=True),indent=2))
    return xyz
