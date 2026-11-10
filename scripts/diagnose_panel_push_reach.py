"""Offline robot-only IK check of recorded sensor panel candidates; no physics.

The recording restores robot encoders/calibration only. All scene geometry is
the saved onboard point cloud. A valid IK pose is not physical success.
"""
import argparse
import json
from itertools import product
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial import cKDTree
from curobo._src.geom.types import Cuboid

from cross_episode_sim.control.observed_arm import robot_calibration
from cross_episode_sim.control.observed_hinged_planner import ObservedHingedPlanner
from cross_episode_sim.control.observed_panel_push import ObservedPanelPush
from cross_episode_sim.control.planner_cache import sensor_arm_planner
from cross_episode_sim.control.sensor_drawer_geometry import merge_occupied, unique_voxels
from cross_episode_sim.control.sensor_hinged import PanelObservation
from cross_episode_sim.finetuning.molmospaces_execution import MolmoSpacesController
from cross_episode_sim.navigation.shaped_docking import RobotGeometry

p=argparse.ArgumentParser();p.add_argument('--opening-run',type=Path,required=True);p.add_argument('--plan-reach',action='store_true')
a=p.parse_args();root=a.opening_run
if (root/'report.json').exists():report=json.loads((root/'report.json').read_text())
else:
    # Interrupted diagnostics still close their state recorder. Recover only
    # the already logged sensor estimates; never substitute fixture truth.
    import ast
    lines=(root.parent/'open.log').read_text().splitlines()
    events=[ast.literal_eval(line) for line in lines if line.startswith("{'phase':")]
    events.append(dict(hinge_estimate=next(ast.literal_eval(line[6:]) for line in lines if line.startswith('HINGE '))))
    report=dict(events=events)
m=mujoco.MjModel.from_binary_path(str(root/'state_recording/model.mjb'));d=mujoco.MjData(m)
with h5py.File(root/'state_recording/states.h5') as f:
    mujoco.mj_setState(m,d,f['states'][-1],mujoco.mjtState.mjSTATE_INTEGRATION)
mujoco.mj_forward(m,d)
c=MolmoSpacesController(m,d);q=c.robot_view.get_move_group('arm').joint_pos.copy()
joints={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt)
        if (m.joint(j).name or '').startswith('robot_0/gripper/')}
geo=RobotGeometry(m,gripper_joint_positions=joints);poses=geo.fk(q)
tcp=poses[geo.tcp_body]@geo.tcp_local;vertices=[];links=[]
for body,geoms in geo.geoms.items():
    name=m.body(body).name or ''
    if not name.startswith('robot_0/gripper/') or name.endswith('/base'):continue
    if geoms:links.append(name[len('robot_0/'):].replace('/','_'))
    for _,local,center,half in geoms:
        T=poses[body]@local;corners=np.array(list(product([-1.,1.],repeat=3)))*half+center
        vertices.extend(corners@T[:3,:3].T+T[:3,3])
pads=(np.asarray(vertices)-tcp[:3,3])@tcp[:3,:3]
saved=np.load(root/'push_candidates.npz');points=saved['points']
estimate=next(e['hinge_estimate'] for e in report['events'] if 'hinge_estimate' in e)
handle=next(e['measured'] for e in reversed(report['events']) if e.get('phase')=='open')
panel=PanelObservation(saved['surfaces'][0],saved['directions'][0],len(points),0.)
candidates=ObservedPanelPush().candidates(points,panel,estimate['center_xy'],handle,
    int(np.sign(estimate['observed_angle'])),pads,edge_margins=(.015,.04,.07),
    oracle_contact_poses=True,preferred_rotation=tcp[:3,:3],use_observed_edge=True,
    height_offsets=(-.12,0.,-.24))
arm=sensor_arm_planner(robot_calibration(m,d,root/'reach_diagnostic_robot'),collision_cache={'cuboid':12000,'mesh':2})
rows=[]
for index,candidate in enumerate(candidates):
    keep=cKDTree(candidate.patch).query(points,distance_upper_bound=.018)[0]>.018
    tangent=np.cross([0.,0.,1.],panel.normal)
    intended=(abs((points-panel.anchor)@panel.normal)<.008)
    intended &= abs((points-candidate.contact[:3,3])@tangent)<.05
    intended &= abs(points[:,2]-candidate.contact[2,3])<.05
    keep &= ~intended
    vox=unique_voxels(np.floor(points[keep]/.01).astype(int))
    boxes=[Cuboid(name=f'observed_{i}',pose=((lo+hi+1)*.005).tolist()+[1.,0.,0.,0.],dims=((hi-lo+1)*.01).tolist())
           for i,(lo,hi) in enumerate(merge_occupied(vox))]
    planner=ObservedHingedPlanner(arm,geo,boxes,finger_geometry=geo,finger_links=links,
        panel_points=np.empty((0,3)),hinge_xy=[0,0],reference_tcp=tcp[:3,3])
    planner.preferred_ik_seeds=[np.array([0.,.25,0.,-1.5,0.,3.35,0.])]
    solution=next(planner.iter_solutions(q,candidate.contact[:3,3],candidate.contact[:3,:3]),None)
    row=dict(index=index,surface=candidate.surface.tolist(),reachable=solution is not None,
             rejection=planner.ik_diagnostics[:1]);rows.append(row);print(json.dumps(row),flush=True)
    if solution is not None:
        if a.plan_reach:
            full=unique_voxels(np.floor(points/.01).astype(int))
            obstacles=[Cuboid(name=f'observed_{i}',pose=((lo+hi+1)*.005).tolist()+[1.,0.,0.,0.],dims=((hi-lo+1)*.01).tolist())
                       for i,(lo,hi) in enumerate(merge_occupied(full))]
            approach=ObservedHingedPlanner(arm,geo,obstacles,finger_geometry=geo,finger_links=links,
                panel_points=np.empty((0,3)),hinge_xy=[0,0],reference_tcp=tcp[:3,3],allow_curobo_fallback=True)
            approach.preferred_ik_seeds=[solution]
            print('PLANNING_REACH',index,flush=True)
            try:
                from cross_episode_sim.control.observed_panel_push import prepare_observed_panel_push
                _,reach,contact=prepare_observed_panel_push(approach,planner,q,candidate,estimate['center_xy'])
            except RuntimeError as exc:
                row['reach_error']=str(exc);print('REACH_REJECTED',index,str(exc),flush=True);continue
            row['reach_validated']=True
            np.savez(root/'panel_transition_plan.npz',reach=reach,contact=contact)
            print('REACH_VALIDATED',index,len(reach),len(contact),flush=True)
        np.savez(root/'reachable_panel_candidate.npz',q=solution,contact=candidate.contact,
                 precontact=candidate.precontact,surface=candidate.surface,patch=candidate.patch)
        break
(root/'panel_reach_diagnostic.json').write_text(json.dumps(dict(
    scope='offline sensor geometry and robot IK only; no physical success',candidates=rows),indent=2))
