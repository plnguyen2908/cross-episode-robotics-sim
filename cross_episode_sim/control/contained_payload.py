"""Collision planning for a container with measured, freely moving contents.

Only scratch planning states project the contents with their container. Live
objects retain their free joints and ordinary contact/friction dynamics.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from cross_episode_sim.control.collision_probe import forward_collision


class ContainedPayloadMixin:
    def select_object(self,obj):
        self._contained_payload=[]
        self._contained_payload_bids=set()
        return super().select_object(obj)

    def kitchen_world_geoms(self):
        carried=getattr(self,'_contained_payload_bids',set())
        return [g for g in super().kitchen_world_geoms()
                if self.model.geom_bodyid[g] not in carried]

    def loaded_payload_bodies(self):
        return self.bread_bids | getattr(self,'_contained_payload_bids',set())

    def loaded_payload_vertices(self):
        from cross_episode_sim.manipulation.edge_access import collision_vertices
        contents=getattr(self,'_contained_payload_bids',set())
        vertices=self.bread_vertices()
        return np.concatenate([vertices,collision_vertices(self.model,self.data,contents)]) if contents else vertices

    def project_payload_contents(self,probe,reference=None):
        reference=self.data if reference is None else reference
        if probe is reference:return
        contents=getattr(self,'_contained_payload',[])
        if not contents:return
        live=reference.body(self.object_name)
        future=probe.body(self.object_name)
        rotation=future.xmat.reshape(3,3) @ live.xmat.reshape(3,3).T
        for name in contents:
            body=reference.body(name)
            joint_id=int(self.model.body_jntadr[self.model.body(name).id])
            address=int(self.model.jnt_qposadr[joint_id])
            probe.qpos[address:address+3]=future.xpos+rotation @ (body.xpos-live.xpos)
            probe.qpos[address+3:address+7]=Rotation.from_matrix(
                rotation @ body.xmat.reshape(3,3)).as_quat(scalar_first=True)
        forward_collision(self.model,probe)

    def navigation_penetration(self,data,carrying):
        contents=getattr(self,'_contained_payload_bids',set())
        if not contents:return super().navigation_penetration(data,carrying)
        if carrying and data is not self.data:
            self.project_payload_contents(data)
        robot,_,object_mask=self.contact_body_masks()
        payload=object_mask.copy();payload[list(contents)]=True
        contacts=data.contact
        a=self.model.geom_bodyid[contacts.geom1];b=self.model.geom_bodyid[contacts.geom2]
        ra,rb,pa,pb=robot[a],robot[b],payload[a],payload[b]
        ma,mb=ra | (carrying & pa),rb | (carrying & pb)
        selected=(ma & ~rb & ~pb) | (mb & ~ra & ~pa)
        selected &= ~(self._contact_floor_mask[a] | self._contact_floor_mask[b])
        return float(np.max(.001-contacts.dist[selected],initial=0.))
