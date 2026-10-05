"""RoboCasa scene data for the unmodified MolmoSpaces TableReorder flow."""
import xml.etree.ElementTree as ET
import cv2
import mujoco
import numpy as np
from cross_episode_sim.manipulation.grasp_qualification import GraspQualification
from cross_episode_sim.controller.manipulation import TableReorder
from molmo_spaces.utils.scene_maps import ProcTHORMap

class RoboCasaLocomanip(GraspQualification):
    # Bind the actual working methods, rather than keeping a second implementation.
    prepare_pickup = TableReorder.prepare_pickup
    transport_payload = TableReorder.transport_payload
    redock_loaded_for_placement = TableReorder.redock_loaded_for_placement
    plan_route = TableReorder.plan_route

    def __init__(self,args,selection):
        # MolmoSpaces tables expose placement sites; add equivalent metadata to
        # the exported RoboCasa fixture without changing collision geometry.
        model=mujoco.MjModel.from_xml_path(args.scene_xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        tree=ET.parse(args.scene_xml)
        corners=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
        for table in selection['tables']:
            bid=model.body(table['body']).id;bids={bid}
            for b in range(model.nbody):
                if model.body_parentid[b] in bids:bids.add(b)
            boxes=[]
            for g in range(model.ngeom):
                if model.geom_bodyid[g] in bids and (model.geom_contype[g] or model.geom_conaffinity[g]):
                    pts=data.geom_xpos[g]+(model.geom_aabb[g,:3]+corners*model.geom_aabb[g,3:])@data.geom_xmat[g].reshape(3,3).T
                    boxes.append((pts.min(0),pts.max(0)))
            low,high=max(boxes,key=lambda box:box[1][2])
            world=(low+high)/2;world[2]=high[2]
            local=data.xmat[bid].reshape(3,3).T@(world-data.xpos[bid])
            name='shared_placement_'+str(bid)
            body=next(b for b in tree.getroot().iter('body') if b.get('name')==table['body'])
            ET.SubElement(body,'site',name=name,type='box',pos=' '.join(map(str,local)),size=f'{(high[0]-low[0])/2} {(high[1]-low[1])/2} .001',rgba='0 0 0 0')
            table['sites']=[name]
        tree.write(args.scene_xml)
        super().__init__(args,selection)
        self.report['locomanip_controller']='TableReorder shared pickup/navigation/transport/placement/redock'

    def _base_nav_map(self):
        if hasattr(self,'_base_map'):return self._base_map
        corners=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
        floors=[];obstacles=[]
        for g in range(self.model.ngeom):
            name=self.model.body(self.model.geom_bodyid[g]).name or ''
            if name.startswith('robot_0/') or self.model.geom_bodyid[g] in self.bread_bids:continue
            if not (self.model.geom_contype[g] or self.model.geom_conaffinity[g]):continue
            pts=self.data.geom_xpos[g]+(self.model.geom_aabb[g,:3]+corners*self.model.geom_aabb[g,3:])@self.data.geom_xmat[g].reshape(3,3).T
            if name.startswith('floor_'):
                if abs(float(pts[:,2].max())) < .05: floors.append((name,pts))
            elif pts[:,2].max()>.05 and pts[:,2].min()<1.8:obstacles.append(pts)
        if not floors:raise RuntimeError('RoboCasa scene has no floor geometry')
        allpoints=np.concatenate([p for _,p in floors]);low=allpoints[:,:2].min(0)-.1;high=allpoints[:,:2].max(0)+.1
        scale=50;shape=tuple(np.ceil((high-low)*scale).astype(int)+1)
        rooms=np.zeros(shape,np.int32);blocked=np.zeros(shape,np.uint8)
        def poly(points):return cv2.convexHull(np.round((points[:,:2]-low)*scale).astype(np.int32)[:,[1,0]])
        names={}
        for i,(name,pts) in enumerate(floors,1):cv2.fillConvexPoly(rooms,poly(pts),i);names[i]=name
        for pts in obstacles:cv2.fillConvexPoly(blocked,poly(pts),1)
        occupancy=(rooms>0)&~blocked.astype(bool)
        world_to_map=np.array([[scale,0,0,-low[0]*scale],[0,scale,0,-low[1]*scale]])
        map_to_world=np.array([[1/scale,0,low[0]],[0,1/scale,low[1]],[0,0,0]])
        self._base_map=ProcTHORMap(occupancy,world_to_map,map_to_world,scale,room_map=rooms,room_ids_to_name=names)
        self._base_map.save(str(self.output/'robocasa_floor_map.png'))
        return self._base_map

    def navigation_penetration(self,data,carrying):
        # RoboCasa uses floor boxes where MolmoSpaces uses planes. Exclude only
        # intended ground contacts; keep all robot/fixture and payload obstacles.
        worst=0.
        for c in data.contact:
            b1,b2=self.model.geom_bodyid[[c.geom1,c.geom2]]
            n1,n2=self.model.body(b1).name or '',self.model.body(b2).name or ''
            if n1.startswith('floor_') or n2.startswith('floor_'):continue
            r1,r2=n1.startswith('robot_0/'),n2.startswith('robot_0/')
            o1=not r1 and b1 not in self.bread_bids;o2=not r2 and b2 not in self.bread_bids
            m1=r1 or (carrying and b1 in self.bread_bids);m2=r2 or (carrying and b2 in self.bread_bids)
            if (m1 and o2) or (m2 and o1):worst=max(worst,-float(c.dist)+.001)
        return worst

    def task_navigate(self,*args,**kwargs):
        result=super().task_navigate(*args,**kwargs)
        self.report['navigation_tested']=True
        return result
