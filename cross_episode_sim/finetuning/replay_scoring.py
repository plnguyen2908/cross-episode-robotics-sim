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
        self.bindings=manifest.get('fixture_bindings',{})
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
        self.regions={f'setting_{i:02d}_{cat}':np.asarray(next(r for r in self.roles if r['role']==f'{cat}_{i:02d}')['destination_position']) for i in range(1,self.config['scene'].get('people',0)+1) for cat in ('bowl','cup')} if self.config['task']=='set_table' else {}
        self.on=False;self.duration={}
        self.rules=duration_rules(self.config['task'],json.loads((source/'initial_snapshot.json').read_text()))
        self.buttons={}
        if 'microwave' in self.bindings:
            self.buttons={k:model.geom('microwave_main_group_'+k+'_button').id for k in ('start','stop')}

    def observe(self,data):
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
            region='default'
            if support in self.surfaces and not support.startswith('burner_'):
                gid=m.geom(self.surfaces[support]['floor_geom']).id
                vertices=self.vertices[r['role']]@body.xmat.reshape(3,3).T+body.xpos
                local=(vertices-data.geom_xpos[gid])@data.geom_xmat[gid].reshape(3,3)
                center,half=m.geom_aabb[gid,:3],m.geom_aabb[gid,3:]
                if np.all(local[:,:2]>=center[:2]-half[:2]-.003) and np.all(local[:,:2]<=center[:2]+half[:2]+.003):region='interior'
            if support=='dining_table':
                matches=[k for k,xy in self.regions.items() if np.linalg.norm(body.xpos[:2]-xy[:2])<.07]
                if len(matches)==1:region=matches[0]
            objects[r['role']]=dict(category=r['category'],support=support,region=region,position=body.xpos.tolist(),quaternion=body.xquat.tolist())
            if 'color' in r:objects[r['role']]['color']=r['color']
        fixtures={}
        for role,b in self.bindings.items():
            if b['kind']=='stove-knob':
                q=float(data.joint(b['joint']).qpos[0])%(2*np.pi)
                fixtures[b['burner']]=dict(category='burner',active=.35<=q<=2*np.pi-.35)
            else:fixtures[role]=dict(closed=all(abs(float(data.joint(x['joint']).qpos[0]))<.03 for x in b.get('leaves',[b])),category={'kitchen_cabinet':'cabinet','cup_drawer':'drawer'}.get(role,role),room='kitchen')
        if 'microwave' in fixtures:fixtures['microwave']['active']=self.on
        return dict(objects=objects,fixtures=fixtures,held=next(iter(held),None),simulation_time=float(data.time),evaluation=self.duration.copy())

    def score(self,initial,data):
        return GOALS[self.config['task']](self.config['scene'],initial,self.snapshot(data))


def load_replay_scene(source):
    from cross_episode_sim.data.export import Layout
    source=Path(source);model=mujoco.MjModel.from_binary_path(str(source/'action_recording/model.mjb'));data=mujoco.MjData(model)
    with np.load(source/'action_recording/initial_state.npz') as a:mujoco.mj_setState(model,data,a['state'],int(a['state_type']))
    mujoco.mj_forward(model,data)
    return model,data,Layout(model,'franka_tidybot'),CompositeReplayScorer(model,data,source)
