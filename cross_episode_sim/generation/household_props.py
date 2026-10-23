"""Simple rigid packaged snacks and open baskets authored by datagen.

These new props have analytic grasp hypotheses, not fabricated qualification
records. Physical success is established only by the episode's atomic checks.
"""
import xml.etree.ElementTree as ET
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation


def add_prop(root, output, role, item, variant):
    scale = 1.08 if variant == 'same_category_alternatives' else 1.
    name = role+'_prop'
    body = ET.SubElement(root.find('worldbody'), 'body', name=name, pos='0 0 1')
    ET.SubElement(body, 'freejoint', name=name+'_joint')
    candidates = []
    def box(key, pos, size, color, mass):
        geom = ET.SubElement(body, 'geom', name=name+'_'+key, type='box',
            pos=' '.join(map(str,np.asarray(pos)*scale)), size=' '.join(map(str,np.asarray(size)*scale)),
            rgba=color, mass=str(mass), friction='1.0 0.01 0.001', group='0')
        visual = dict(geom.attrib, name=name+'_'+key+'_visual', contype='0', conaffinity='0', group='1', mass='0')
        ET.SubElement(body, 'geom', **visual)
    if item['category'] == 'snack_box':
        box('package', [0,0,.04], [.025,.035,.04], '.85 .48 .10 1', .08)
        for yaw in (0,90,180,270):
            pose = np.eye(4)
            pose[:3,:3] = Rotation.from_euler('z',yaw,degrees=True).as_matrix() @ np.diag([1.,-1.,-1.])
            pose[:3,3] = np.array([0,0,.057])*scale
            candidates.append(pose)
    else:
        color = {'purple': '.30 .18 .42 1', 'orange': '.85 .48 .10 1',
                 'blue': '.12 .3 .8 1', 'yellow': '.85 .72 .12 1'}[item['color']]
        box('floor', [0,0,.008], [.12,.14,.008], color, .10)
        for sign in (-1,1):
            box('side_x'+str(sign), [sign*.115,0,.05], [.005,.14,.05], color, .025)
            box('side_y'+str(sign), [0,sign*.135,.05], [.11,.005,.05], color, .025)
            # Exterior tabs give top-down finger access without reaching across
            # the basket opening or trapping a finger among its contents.
            box('handle'+str(sign), [sign*.16,0,.08], [.045,.025,.012], color, .015)
            for depth in (.075,.085):
                pose=np.eye(4)
                pose[:3,:3]=np.diag([1.,-1.,-1.])
                pose[:3,3]=np.array([sign*.18,0,depth])*scale
                candidates.append(pose)
                # Keep the fingers on the exterior tab while allowing the
                # wrist to lean away from a high counter or overhead cabinet.
                # These are hypotheses; normal swept collision/grasp checks
                # must qualify them before carrying a loaded basket.
                for angle in (-20., 20.):
                    leaned=pose.copy()
                    leaned[:3,:3]=Rotation.from_euler('x',angle,degrees=True).as_matrix() @ pose[:3,:3]
                    candidates.append(leaned)
    path=output/(role+'_grasps.npz')
    np.savez_compressed(path,transforms=np.array(candidates))
    model_path=output/(role+'_asset.xml')
    asset_root=ET.Element('mujoco')
    asset_body=deepcopy(body);asset_body.set('pos','0 0 0')
    ET.SubElement(asset_root,'worldbody').append(asset_body)
    ET.ElementTree(asset_root).write(model_path)
    return dict(body=name,asset='authored/'+item['category'],key=name,
        grasp_path=str(path.resolve()),setup={'object_scale':[scale]*3,'source':'authored',
            'asset':'authored/'+item['category'],'model_xml':str(model_path.resolve())},recording=str(output),
        evidence=[],annotation_policy='analytic hypotheses; requires physical validation')
