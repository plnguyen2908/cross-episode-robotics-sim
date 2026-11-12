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
        self.initial_heights={r:float(data.xpos[r,2]) for r in self.roots}
        self.initial_appliance_objects={key:set() for key in ('fridge','microwave')}
        for contact in data.contact:
            if abs(contact.frame[2])<.7:continue
            for a,b in ((contact.geom1,contact.geom2),(contact.geom2,contact.geom1)):
                root=int(self.lookup[model.geom_bodyid[a]])
                if root<0:continue
                body=int(model.geom_bodyid[b]);names=[model.geom(b).name or '']
                while body>0:
                    names.append(model.body(body).name or '');body=int(model.body_parentid[body])
                for key in self.initial_appliance_objects:
                    if key in ' '.join(names).lower():self.initial_appliance_objects[key].add(root)


    def touching_hand(self,data):
        result=set();m=self.model
        for c in data.contact:
            a,b=map(int,m.geom_bodyid[[c.geom1,c.geom2]])
            if a in self.gripper and self.lookup[b]>=0:result.add(int(self.lookup[b]))
            if b in self.gripper and self.lookup[a]>=0:result.add(int(self.lookup[a]))
        return result

    def pick_from(self,data,source):
        expected=self.initial_appliance_objects[source]
        held=self.touching_hand(data)
        lifted={r for r in expected if float(data.xpos[r,2])-self.initial_heights[r]>.08}
        correct=held & lifted & expected
        return dict(success=bool(correct),source=source,initial_source_objects=sorted(expected),
                    held_source_objects=sorted(correct))

    def placement(self,data,case,*,basket='purple basket'):
        m=self.model
        token={'place':'dining_table','cup_in_drawer':'drawer',
               'bowl_in_cabinet':'cabinet','pot_on_stove':'stove','place_microwave':'microwave','grocery_in_basket':'purple_basket'}[case]
        if case=='grocery_in_basket':
            if basket not in ('purple basket','orange basket'):raise ValueError('Unknown basket category')
            token=basket.replace(' ','_')
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
                if case=='grocery_in_basket':
                    # Interior support, not a bottle balanced on the basket rim.
                    if not (m.geom(b).name or '').endswith('_floor'):continue
                    geoms=[i for i in range(m.ngeom) if self.lookup[m.geom_bodyid[i]]==root
                           and (m.geom_contype[i] or m.geom_conaffinity[i])]
                    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
                    vertices=[]
                    for i in geoms:
                        if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH:
                            mesh=m.geom_dataid[i];a0=m.mesh_vertadr[mesh];n=m.mesh_vertnum[mesh];v=m.mesh_vert[a0:a0+n]
                        else:v=m.geom_aabb[i,:3]+corners*m.geom_aabb[i,3:]
                        vertices.extend(v@data.geom_xmat[i].reshape(3,3).T+data.geom_xpos[i])
                    local_vertices=(np.asarray(vertices)-data.geom_xpos[b])@data.geom_xmat[b].reshape(3,3)
                    center,half=m.geom_aabb[b,:3],m.geom_aabb[b,3:]
                    if not (np.all(local_vertices[:,:2]>=center[:2]-half[:2]-.003)
                            and np.all(local_vertices[:,:2]<=center[:2]+half[:2]+.003)):continue
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
