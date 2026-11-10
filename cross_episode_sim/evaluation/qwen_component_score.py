"""Private physical component scoring. Never import this into an execution tool."""
import numpy as np
import mujoco
from .grasp_contact import contact_root_lookup


class ComponentScore:
    def __init__(self, model, data):
        self.model=model
        self.roots=[int(model.jnt_bodyid[j]) for j in range(model.njnt)
                    if model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]
        self.lookup=contact_root_lookup(model.body_parentid,self.roots)
        self.gripper={i for i in range(model.nbody)
                      if (model.body(i).name or '').startswith('robot_0/gripper/')}
        self.initially_held=self.touching_hand(data)
        self.initial=data.qpos.copy()

    def touching_hand(self,data):
        result=set();m=self.model
        for c in data.contact:
            a,b=map(int,m.geom_bodyid[[c.geom1,c.geom2]])
            if a in self.gripper and self.lookup[b]>=0:result.add(int(self.lookup[b]))
            if b in self.gripper and self.lookup[a]>=0:result.add(int(self.lookup[a]))
        return result

    def placement(self,data,case):
        m=self.model
        token={'place':'dining_table','cup_in_drawer':'drawer',
               'bowl_in_cabinet':'cabinet','pot_on_stove':'stove'}[case]
        # RoboCasa stack names encode the fixture through its joint suffix,
        # not necessarily a literal "cabinet"/"drawer" in every shelf name.
        suffix={'drawer':'_slidejoint','cabinet':'_doorhinge'}.get(token)
        prefixes=[(m.joint(j).name or '')[:-len(suffix)] for j in range(m.njnt)
                  if suffix and (m.joint(j).name or '').endswith(suffix)]
        supported=set()
        for c in data.contact:
            if abs(c.frame[2])<.7 or c.dist>.002:continue
            for a,b in ((c.geom1,c.geom2),(c.geom2,c.geom1)):
                root=int(self.lookup[m.geom_bodyid[a]])
                if root not in self.initially_held:continue
                body=int(m.geom_bodyid[b]);names=[m.geom(b).name or '']
                while body>0:
                    names.append(m.body(body).name or '');body=int(m.body_parentid[body])
                if token not in ' '.join(names).lower() and not any(
                        name.startswith(prefix) for name in names for prefix in prefixes):continue
                # The object's center must lie over the supporting surface,
                # rather than leaning on the outside of a drawer/cabinet.
                center=m.geom_aabb[b,:3];half=m.geom_aabb[b,3:]
                local=(data.body(root).xpos-data.geom_xpos[b])@data.geom_xmat[b].reshape(3,3)
                if np.all(np.abs(local[:2]-center[:2])<=half[:2]+.003):supported.add(root)
        free=supported-self.touching_hand(data)
        if case=='pot_on_stove':
            sites=[i for i in range(m.nsite) if 'burner_front_left_place_site' in (m.site(i).name or '')]
            free={r for r in free if any(np.linalg.norm(data.body(r).xpos[:2]-data.site(s).xpos[:2])<.09 for s in sites)}
        return dict(success=bool(free),initially_held=list(self.initially_held),supported=list(supported),released_supported=list(free))

    def pot_knob(self,data):
        m=self.model
        sites={label:next((i for i in range(m.nsite) if f'burner_{label}_place_site' in (m.site(i).name or '')),None)
               for label in ('front_left','front_right','rear_left','rear_right','center')}
        matched=set()
        for body in self.roots:
            if not any(w in (m.body(body).name or '').lower() for w in ('pot','pan','cookware')):continue
            choices=[(np.linalg.norm(data.body(body).xpos-data.site(i).xpos),label) for label,i in sites.items() if i is not None]
            if choices and min(choices)[0]<.18:matched.add(min(choices)[1])
        requested=[j for j in range(m.njnt) if any('knob_'+label+'_joint' in (m.joint(j).name or '') for label in matched)]
        changed=[]
        for j in range(m.njnt):
            if 'knob_' not in (m.joint(j).name or ''):continue
            address=m.jnt_qposadr[j]
            if abs(data.qpos[address]-self.initial[address])>.1:changed.append(j)
        from ..fixtures.stove_knob import burner_on
        success=bool(len(requested)==1 and set(changed)==set(requested)
                     and burner_on(float(data.qpos[m.jnt_qposadr[requested[0]]])))
        return dict(success=success,pot_burners=sorted(matched),changed_knobs=[m.joint(j).name for j in changed])
