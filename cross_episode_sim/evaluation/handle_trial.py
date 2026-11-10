"""Private post-execution handle scoring. Never import into a policy/controller."""
import numpy as np
import mujoco


def score_handle_trial(model, initial_qpos, verification_states, *, full_open=False, pull_distance=0.):
    data=mujoco.MjData(model)
    pads=[model.body(n).id for n in ['robot_0/gripper/left_pad','robot_0/gripper/right_pad']]
    checks=[];handles=set()
    for state in verification_states:
        mujoco.mj_setState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model,data)
        touch=[set(),set()]
        for i,c in enumerate(data.contact):
            force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
            if force[0]<=0:continue
            for g1,g2 in [(c.geom1,c.geom2),(c.geom2,c.geom1)]:
                b=int(model.geom_bodyid[g1]);name=model.geom(g2).name or '';body=model.body(int(model.geom_bodyid[g2])).name or ''
                if b in pads and 'handle' in (name+' '+body).lower():touch[pads.index(b)].add(body)
        common=touch[0]&touch[1];checks.append(bool(common));handles.update(common)
    relevant=set()
    for name in handles:
        body=model.body(name).id
        while body:
            relevant.update(j for j in range(model.njnt) if model.jnt_bodyid[j]==body and model.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE)
            body=int(model.body_parentid[body])
    slides={}
    for j in relevant:
        adr=model.jnt_qposadr[j];initial=float(initial_qpos[adr]);position=float(data.qpos[adr])
        # Opening endpoint is the endpoint furthest from the starting position.
        endpoint=float(max(model.jnt_range[j],key=lambda limit:abs(limit-initial)))
        slides[model.joint(j).name]=dict(position=position,initial=initial,displacement=position-initial,limits=model.jnt_range[j].tolist(),remaining_to_open_stop=abs(position-endpoint))
    moved=max([abs(v['displacement']) for v in slides.values()]+[0.])
    opened=any(abs(v['displacement'])>.05 and v['remaining_to_open_stop']<.01 for v in slides.values())
    success=bool(checks) and all(checks) and (opened if full_open else (pull_distance<=0 or moved>=.8*pull_distance))
    return dict(success=success,private_both_pad_handle_contact=checks,private_full_open=slides)
