"""Private post-execution contact audit; never an execution-policy input."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import h5py,mujoco,numpy as np

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
report=json.loads((a.run/'report.json').read_text())
m=mujoco.MjModel.from_binary_path(str(a.run/'state_recording/model.mjb'));d=mujoco.MjData(m)
roots={int(m.jnt_bodyid[m.joint(name).id]) for name in report['private_final']}
def target_body(body):
    while body:
        if body in roots:return True
        body=int(m.body_parentid[body])
    return False
counts=defaultdict(int);maximum=defaultdict(float);other_contacts={};frames=0
base_ids=[m.joint('robot_0/'+name).id for name in ('base_x','base_y','base_theta')]
base_addresses=m.jnt_qposadr[base_ids];initial_base=None;precontact_max_translation=0.;precontact_max_yaw=0.;contact_started=False
with h5py.File(a.run/'state_recording/states.h5') as f:
    for state,phase in zip(f['states'],f['phases']):
        phase=phase.decode() if isinstance(phase,bytes) else str(phase)
        mujoco.mj_setState(m,d,state,mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
        base=d.qpos[base_addresses].copy()
        if initial_base is None:initial_base=base
        contact_started |= phase=='push_contact'
        if not contact_started:
            precontact_max_translation=max(precontact_max_translation,float(np.linalg.norm(base[:2]-initial_base[:2])))
            yaw=base[2]-initial_base[2]
            precontact_max_yaw=max(precontact_max_yaw,abs(float(np.arctan2(np.sin(yaw),np.cos(yaw)))))
        if 'push' not in phase:continue
        frames+=1
        for i,contact in enumerate(d.contact):
            for ga,gb in ((contact.geom1,contact.geom2),(contact.geom2,contact.geom1)):
                body=int(m.geom_bodyid[ga]);other=int(m.geom_bodyid[gb]);name=m.body(body).name or ''
                if not name.startswith('robot_0/') or (m.body(other).name or '').startswith('robot_0/'):continue
                if not target_body(other):
                    force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
                    if force[0]>0:
                        key=name+' -> '+(m.body(other).name or m.geom(gb).name or str(other))
                        entry=other_contacts.setdefault(key,dict(samples=0,max_normal_force_n=0.))
                        entry['samples']+=1;entry['max_normal_force_n']=max(entry['max_normal_force_n'],float(force[0]))
                    continue
                force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
                if force[0]<=0:continue
                counts[name]+=1;maximum[name]=max(maximum[name],float(force[0]))
result=dict(scope='private sampled-state post-execution audit',sampled_frames=frames,
            base_motion_before_panel_contact=dict(max_translation_m=precontact_max_translation,max_yaw_radians=precontact_max_yaw,contact_phase_reached=contact_started),
            non_target_contacts=other_contacts,contact_samples_by_body=dict(counts),max_normal_force_n_by_body=dict(maximum),
            non_gripper_contact_bodies=[n for n in counts if not n.startswith('robot_0/gripper/')])
(a.run/'private_push_contact_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
