"""Measured composite replay scoring; no saved final labels enter execution."""
import json
from pathlib import Path
import numpy as np
import mujoco
from cross_episode_sim.evaluation.composite_goals import GOALS
from cross_episode_sim.evaluation.heating_duration import duration_rules, accumulate_duration


def descendants(model, name):
    root=model.body(name).id; ids={root}
    for i in range(root+1,model.nbody):
        if model.body_parentid[i] in ids:ids.add(i)
    return ids


class CompositeReplayScorer:
    def __init__(self, model, data, source):
        self.model=model
        self.config=json.loads((source/'config.json').read_text())
        manifest=json.loads((source/'task_manifest.json').read_text())
        self.roles=manifest['bindings'];self.objects={r['role']:descendants(model,r['body']) for r in self.roles}
        self.surfaces=dict(manifest.get('surfaces',{}))
        self.supports={('kitchen_counter' if k=='kitchen' else 'dining_table' if k=='dining' else k):descendants(model,v) for k,v in manifest['supports'].items()}
        for name,spec in self.surfaces.items():self.supports[name]=descendants(model,spec['body'])
        for r in self.roles:
            if r['category']=='basket':
                self.supports[r['role']]=self.objects[r['role']]
                self.surfaces[r['role']]=dict(body=r['body'],floor_geom=r['body']+'_floor')
        self.baskets=[r['role'] for r in self.roles if r['category']=='basket']
        self.bindings=manifest.get('fixture_bindings',{})
        # Any articulated kitchen cabinet shelf or drawer is an eligible store
        # (user decision, November 16, 2026); the bound one keeps its role name.
        from cross_episode_sim.generation.household_bindings import storage_units
        bound={self.surfaces[k]['body'] for k in self.surfaces}
        self.storage={k:u for k,u in storage_units(model).items() if u['support'] not in bound}
        for key,unit in self.storage.items():
            self.supports[key]=descendants(model,unit['support'])
            self.surfaces[key]=dict(body=unit['support'],floor_geom=unit['floor_geom'])
        self._active_burner_joints=[b['joint'] for b in self.bindings.values() if b['kind']=='stove-knob']
        self.gripper={i for i in range(model.nbody) if (model.body(i).name or '').startswith('robot_0/gripper/')}
        self.vertices={}
        corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        for r in self.roles:
            points=[];body=data.body(r['body'])
            for g in range(model.ngeom):
                if model.geom_bodyid[g] not in self.objects[r['role']] or not (model.geom_contype[g] or model.geom_conaffinity[g]):continue
                if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                    mesh=model.geom_dataid[g];a=model.mesh_vertadr[mesh];n=model.mesh_vertnum[mesh];local=model.mesh_vert[a:a+n]
                else:local=model.geom_aabb[g,:3]+corners*model.geom_aabb[g,3:]
                points.extend(local@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g])
            self.vertices[r['role']]=(np.asarray(points)-body.xpos)@body.xmat.reshape(3,3)
        self.on=False;self.duration={};self.opened=set()
        self.rules=duration_rules(self.config['task'],json.loads((source/'initial_snapshot.json').read_text()))
        self.buttons={}
        if 'microwave' in self.bindings:
            self.buttons={k:model.geom('microwave_main_group_'+k+'_button').id for k in ('start','stop')}

    def observe(self,data):
        # Private record of every door or drawer opened during the episode, for
        # the progress count only (a fixture closed at the end earns progress
        # only if it had been opened).
        for role,b in self.bindings.items():
            if b['kind']!='stove-knob' and role not in self.opened and any(
                    abs(float(data.joint(x['joint']).qpos[0]))>=.03 for x in b.get('leaves',[b])):self.opened.add(role)
        if self.buttons:
            touched=set()
            for c in data.contact:
                for key,gid in self.buttons.items():
                    if gid in (c.geom1,c.geom2):
                        other=c.geom2 if c.geom1==gid else c.geom1
                        if self.model.geom_bodyid[other] in self.gripper:touched.add(key)
            opened=abs(float(data.joint(self.bindings['microwave']['joint']).qpos[0]))>.03
            if opened or (self.on and 'stop' in touched):self.on=False
            elif not self.on and 'start' in touched:self.on=True
        if self.rules:
            active=self.on or any(.35 <= float(data.joint(j).qpos[0])%(2*np.pi) <= 2*np.pi-.35 for j in self._active_burner_joints)
            # No supported object can heat while every appliance is off. Still
            # account at every physics boundary, including the turn-off edge.
            snapshot=self.snapshot(data) if active else dict(objects={},fixtures={},held=None)
            accumulate_duration(self.duration,self.rules,float(data.time),snapshot)

    def snapshot(self,data):
        objects={};held=[];m=self.model
        for r in self.roles:
            bodies=self.objects[r['role']];supports=set();touch=False
            for c in data.contact:
                a,b=m.geom_bodyid[[c.geom1,c.geom2]]
                if a in bodies:other,normal=b,-c.frame[:3]
                elif b in bodies:other,normal=a,c.frame[:3]
                else:continue
                if other in self.gripper and c.dist<=.002:touch=True
                if normal[2]>.7 and c.dist<=.002:supports.update(k for k,ids in self.supports.items() if other in ids)
            body=data.body(r['body'])
            if touch:held.append(r['role']);supports=set()
            # Both burners share one physical stove body. Resolve by native site.
            if any(k.startswith('burner_') for k in supports):
                supports={k for k in supports if not k.startswith('burner_')}|{k for k,v in self.surfaces.items() if v.get('burner_site') and np.linalg.norm(body.xpos[:2]-data.site(v['burner_site']).xpos[:2])<.09}
            support=next(iter(supports)) if len(supports)==1 else None
            # Baskets are judged below; other open surfaces keep the footprint rule.
            region='interior' if (support in self.surfaces and not support.startswith('burner_')
                                  and support not in self.baskets and self._over_floor(data,r,support)) else 'default'
            objects[r['role']]=dict(category=r['category'],support=support,region=region,position=body.xpos.tolist(),quaternion=body.xquat.tolist())
            if 'color' in r:objects[r['role']]['color']=r['color']
        # In a basket (user decision, November 13): a released item that touches
        # an inner face of the basket, or touches an item already counted as in
        # it. It may lean on a wall and reach past the rim. An item that also
        # stands on something outside the basket is not in it.
        inside={}
        for _ in self.roles:
            changed=False
            for r in self.roles:
                role=r['role']
                if role in inside or role in held or r['category']=='basket':continue
                for basket in self.baskets:
                    contents=set().union(*[self.objects[k] for k,v in inside.items() if v==basket])
                    if not (self._touches_inside(data,self.objects[role],basket) or self._touches(data,self.objects[role],contents)):continue
                    if objects[role]['support']!=basket and not self._at_rest(data,r):continue
                    if self._stands_on_other(data,self.objects[role],self.objects[basket]|contents):continue
                    inside[role]=basket;objects[role].update(support=basket,region='interior');changed=True;break
            if not changed:break
        fixtures={}
        for role,b in self.bindings.items():
            if b['kind']=='stove-knob':
                q=float(data.joint(b['joint']).qpos[0])%(2*np.pi)
                fixtures[b['burner']]=dict(category='burner',active=.35<=q<=2*np.pi-.35)
            else:fixtures[role]=dict(closed=all(abs(float(data.joint(x['joint']).qpos[0]))<.03 for x in b.get('leaves',[b])),category={'kitchen_cabinet':'cabinet','cup_drawer':'drawer'}.get(role,role),room='kitchen')
        for key,unit in self.storage.items():
            fixtures[key]=dict(closed=all(abs(float(data.joint(j).qpos[0]))<.03 for j in unit['joints']),category=unit['category'],room='kitchen')
        if 'microwave' in fixtures:fixtures['microwave']['active']=self.on
        evaluation=self.duration.copy();evaluation['opened_fixtures']=sorted(self.opened)
        return dict(objects=objects,fixtures=fixtures,held=next(iter(held),None),simulation_time=float(data.time),evaluation=evaluation)

    def _over_floor(self,data,r,support):
        m=self.model;body=data.body(r['body']);gid=m.geom(self.surfaces[support]['floor_geom']).id
        vertices=self.vertices[r['role']]@body.xmat.reshape(3,3).T+body.xpos
        local=(vertices-data.geom_xpos[gid])@data.geom_xmat[gid].reshape(3,3)
        center,half=m.geom_aabb[gid,:3],m.geom_aabb[gid,3:]
        return bool(np.all(local[:,:2]>=center[:2]-half[:2]-.003) and np.all(local[:,:2]<=center[:2]+half[:2]+.003))

    def _contacts(self,data,bodies):
        """Yield (other body, other geom, point, unit normal toward the item) for touching contacts."""
        m=self.model
        for c in data.contact:
            if c.dist>.002:continue
            a,b=map(int,m.geom_bodyid[[c.geom1,c.geom2]])
            if (a in bodies)==(b in bodies):continue
            if a in bodies:yield b,int(c.geom2),c.pos,-c.frame[:3]
            else:yield a,int(c.geom1),c.pos,c.frame[:3]

    def _touches_inside(self,data,bodies,basket):
        """Contact with the top of the basket floor or an inward-facing wall surface."""
        m=self.model;gid=m.geom(self.surfaces[basket]['floor_geom']).id;frame=data.geom_xmat[gid].reshape(3,3)
        center,half=m.geom_aabb[gid,:3],m.geom_aabb[gid,3:]
        for other,geom,point,normal in self._contacts(data,bodies):
            if other not in self.objects[basket]:continue
            local=(point-data.geom_xpos[gid])@frame-center;normal=normal@frame
            if np.any(np.abs(local[:2])>half[:2]+.003):continue
            if geom==gid:
                if normal[2]>.5:return True
            elif -(local[:2]@normal[:2])>.2*np.linalg.norm(local[:2]):return True
        return False

    def _touches(self,data,bodies,others):
        return bool(others) and any(other in others for other,*_ in self._contacts(data,bodies))

    def _stands_on_other(self,data,bodies,allowed):
        return any(normal[2]>.7 and other not in allowed and other not in self.gripper for other,_,_,normal in self._contacts(data,bodies))

    def _at_rest(self,data,r):
        m=self.model;joint=int(m.body_jntadr[m.body(r['body']).id])
        if joint<0 or m.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_FREE:return False
        adr=int(m.jnt_dofadr[joint])
        return bool(np.linalg.norm(data.qvel[adr:adr+3])<.03 and np.linalg.norm(data.qvel[adr+3:adr+6])<.1)

    def score(self,initial,data):
        return GOALS[self.config['task']](self.config['scene'],initial,self.snapshot(data))

    def progress(self,initial,data):
        from cross_episode_sim.evaluation.composite_goals import goal_progress
        return goal_progress(self.config['task'],self.config['scene'],initial,self.snapshot(data))


def load_replay_scene(source):
    from cross_episode_sim.data.export import Layout
    # Same scene the policy ran in: all fixture joints live (rebuilt from scene.xml
    # when the saved model froze them), so saved rollout states load onto it.
    from cross_episode_sim.evaluation.articulated_scene import load_saved_scene
    source=Path(source);model,data,_=load_saved_scene(source)
    return model,data,Layout(model,'franka_tidybot'),CompositeReplayScorer(model,data,source)
