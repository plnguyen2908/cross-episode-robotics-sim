"""Collision planning with an RGB-D-observed moving panel, not fixture truth."""
import numpy as np
from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner


class ObservedHingedPlanner(ObservedGraspPlanner):
    def cspace_search(self,goal,current):
        # The base is fixed in an arm-only graph search. Its calibrated boxes
        # have the same world clearance at every arm configuration; validate
        # them once rather than rejecting the entire search on covering-sphere
        # overhang. Arm/mast/finger geometry remains in the native search.
        if len(self.panel_points):
            raise RuntimeError('Global arm search cannot move a contacted panel')
        for box in self.base_boxes:
            if not self.world_clear_box(box):
                raise RuntimeError('Fixed base overlaps observed world')
        cfg=self.arm.planner.kinematics.config.kinematics_config
        ids=cfg.get_sphere_index_from_link_name('base')
        saved=cfg.link_spheres[:,ids,:].clone()
        try:
            cfg.disable_link_spheres('base')
            return super().cspace_search(goal,current)
        finally:
            # Restore before the caller validates the returned trajectory with
            # full self-collision and exact base boxes, even if planning raises.
            cfg.link_spheres[:,ids,:]=saved

    def __init__(self,*args,panel_points,hinge_xy,reference_tcp,finger_geometry=None,finger_links=(),**kwargs):
        super().__init__(*args,**kwargs)
        self.panel_points=np.asarray(panel_points).copy();self.panel_points.setflags(write=False)
        self.hinge_xy=np.asarray(hinge_xy)
        self.reference_tcp=np.asarray(reference_tcp)
        cfg=self.arm.planner.kinematics.config.kinematics_config
        self.base_sphere_ids=cfg.get_sphere_index_from_link_name('base')
        self.finger_geometry=finger_geometry
        self.finger_sphere_ids=np.concatenate([cfg.get_sphere_index_from_link_name(n).cpu().numpy() for n in finger_links]) if finger_links else np.array([],dtype=int)
        self._checked_q=None
        self.base_boxes=[b for b in self.geometry.boxes(np.zeros(7)) if b.name.startswith('robot_0/base/')]
        if not self.base_boxes:raise ValueError('Missing calibrated base collision shapes')

    def world_clear(self, spheres, margin=.005, base_offset=None, base_yaw=0.):
        """Use calibrated base OBBs for static as well as moving obstacles."""
        if getattr(self,'ik_only',False):return True      # user rule: IK found is enough (2026-11-18)
        from dataclasses import replace
        checked=np.asarray(spheres).copy()
        checked[:,self.base_sphere_ids.cpu().numpy(),3]=-1
        finger_geometry=getattr(self,'finger_geometry',None)
        if finger_geometry is not None and self._checked_q is not None:
            checked[:,self.finger_sphere_ids,3]=-1
        if not super().world_clear(checked,margin):return False
        offset=np.zeros(3) if base_offset is None else np.asarray(base_offset)
        co,si=np.cos(base_yaw),np.sin(base_yaw);base_rotation=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
        exact_boxes=list(self.base_boxes)
        if finger_geometry is not None and self._checked_q is not None:
            for sample,q in enumerate(np.atleast_2d(self._checked_q)):
                # Covering spheres are a conservative broad phase. Only use
                # calibrated boxes where those spheres actually overlap.
                fingers=np.asarray(spheres)[sample:sample+1,self.finger_sphere_ids,:]
                if super().world_clear(fingers,margin):continue
                exact_boxes.extend(b for b in finger_geometry.boxes(q) if b.name.startswith('robot_0/gripper/') and '/base/' not in b.name)
        for box in exact_boxes:
            if not self.world_clear_box(replace(box,center=base_rotation@box.center+offset,axes=base_rotation@box.axes),margin):
                self.last_rejection=dict(self.last_rejection,kind='observed_static_base_box',part=box.name)
                return False
        return True

    def clear(self,positions,margin=.005):
        qs=np.atleast_2d(positions)
        if len(qs)>32:return all(self.clear(qs[i:i+32],margin) for i in range(0,len(qs),32))
        self._checked_q=qs
        if not super().clear(qs,margin):return False
        if not len(self.panel_points):return True
        # The original calibrated collision spheres remain in use. Batch the
        # observed moving-panel distances on GPU instead of rebuilding worlds.
        import torch
        from curobo.types import JointState
        state=JointState.from_position(torch.as_tensor(qs,dtype=torch.float32,device='cuda'),joint_names=self.arm.names)
        tcps=np.array([(self.geometry.fk(q)[self.geometry.tcp_body]@self.geometry.tcp_local)[:3,3] for q in qs])
        a=self.reference_tcp[:2]-self.hinge_xy;b=tcps[:,:2]-self.hinge_xy
        angles=np.arctan2(a[0]*b[:,1]-a[1]*b[:,0],b@a)
        with torch.no_grad():
            spheres=self.arm.planner.kinematics.compute_kinematics(state).robot_spheres.reshape(len(qs),-1,4)
            center=torch.as_tensor(np.r_[self.hinge_xy,0.],device='cuda',dtype=torch.float32)
            cloud=torch.as_tensor(self.panel_points,device='cuda',dtype=torch.float32)-center
            theta=torch.as_tensor(angles,device='cuda',dtype=torch.float32);co=theta.cos()[:,None];si=theta.sin()[:,None]
            moved=cloud[None].expand(len(qs),-1,-1).clone()
            moved[:,:,0]=co*cloud[None,:,0]-si*cloud[None,:,1]
            moved[:,:,1]=si*cloud[None,:,0]+co*cloud[None,:,1];moved+=center
            gap=torch.cdist(spheres[:,:,:3],moved)-spheres[:,:,3,None]-.004
            # Use calibrated base boxes for narrow phase instead of oversized
            # covering spheres. Arm and mast spheres remain unchanged.
            gap[:,self.base_sphere_ids,:]=float('inf')
            if self.finger_geometry is not None:
                gap[:,self.finger_sphere_ids,:]=float('inf')
                for sample,q in enumerate(qs):
                    for box in self.finger_geometry.boxes(q):
                        if not box.name.startswith('robot_0/gripper/') or '/base/' in box.name:continue
                        local=(moved[sample]-torch.as_tensor(box.center,device='cuda',dtype=torch.float32))@torch.as_tensor(box.axes,device='cuda',dtype=torch.float32)
                        delta=local.abs()-torch.as_tensor(box.half,device='cuda',dtype=torch.float32)
                        signed=delta.clamp_min(0).norm(dim=-1)+delta.max(dim=-1).values.clamp_max(0)-.004
                        if float(signed.min())<margin:
                            self.last_rejection={'kind':'observed_panel_finger_box','gap':float(signed.min()),'part':box.name}
                            return False
            for box in self.base_boxes:
                local=(moved-torch.as_tensor(box.center,device='cuda',dtype=torch.float32))@torch.as_tensor(box.axes,device='cuda',dtype=torch.float32)
                delta=local.abs()-torch.as_tensor(box.half,device='cuda',dtype=torch.float32)
                signed=delta.clamp_min(0).norm(dim=-1)+delta.max(dim=-1).values.clamp_max(0)-.004
                if float(signed.min())<margin:
                    self.last_rejection={'kind':'observed_panel_base_box','gap':float(signed.min()),'part':box.name}
                    return False
            minimum=float(gap.min())
        if minimum<margin:
            flat_index=int(gap.argmin()); sample,sphere,point=np.unravel_index(flat_index,tuple(gap.shape))
            self.last_rejection={'kind':'observed_moving_panel_collision','gap':minimum,
                'sphere':spheres[sample,sphere].cpu().tolist(),
                'point':moved[sample,point].cpu().tolist()}
            return False
        return True


    def panel_nearest(self, queries, panel_motion=None):
        """Exact nearest distances against one rigid observed panel."""
        from scipy.spatial import cKDTree
        original=self.panel_points
        if getattr(self,'_panel_tree_source',None) is not original:
            self._panel_tree=cKDTree(original)
            self._panel_tree_source=original
        if panel_motion is not None:
            rotation,center=panel_motion
            queries=(np.asarray(queries)-center)@rotation+center
        return self._panel_tree.query(queries)

    def clear_contact_retreat(self, q, offset, margin=.005, base_yaw=0., panel_motion=None):
        """Check a coordinated base/arm pose against the stationary held leaf."""
        import torch
        from curobo.types import JointState
        state=JointState.from_position(torch.as_tensor(np.asarray(q)[None],dtype=torch.float32,device='cuda'),joint_names=self.arm.names)
        with torch.no_grad():
            spheres=self.arm.planner.kinematics.compute_kinematics(state).robot_spheres.reshape(1,-1,4)
            cfg=self.arm.planner.kinematics.get_self_collision_config();pairs=cfg.collision_pairs.long()
            radii=spheres[...,3]+cfg.sphere_padding
            gaps=(spheres[:,pairs[:,0],:3]-spheres[:,pairs[:,1],:3]).norm(dim=-1)-radii[:,pairs[:,0]]-radii[:,pairs[:,1]]
            if float(gaps.min())<margin:
                self.last_rejection={'kind':'contact_self_collision','gap':float(gaps.min())};return False
            co,si=np.cos(base_yaw),np.sin(base_yaw);base_rotation=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
            spheres=spheres.cpu().numpy();spheres[:,:,:3]=spheres[:,:,:3]@base_rotation.T+offset
        # Keep ordinary scene collision checks, including the base.
        self._checked_q=np.asarray(q)[None]
        if not self.world_clear(spheres,margin,base_offset=offset,base_yaw=base_yaw):return False
        # Reuse the reference panel index; rigid motion preserves distance.
        original=self.panel_points
        panel=original
        if panel_motion is not None:
            rotation,center=panel_motion
            panel=(original-center)@rotation.T+center
        moving=spheres[0].copy();moving[self.base_sphere_ids.cpu().numpy(),3]=-1
        exact_panel_boxes=list(self.base_boxes)
        if self.finger_geometry is not None:
            moving[self.finger_sphere_ids,3]=-1
            exact_panel_boxes.extend(b for b in self.finger_geometry.boxes(q) if b.name.startswith('robot_0/gripper/') and '/base/' not in b.name)
        moving=moving[moving[:,3]>0]
        if len(self.panel_points) and len(moving):
            distance,nearest=self.panel_nearest(moving[:,:3],panel_motion)
            distances=distance-moving[:,3]-.004
            if distances.min()<margin:
                index=int(np.argmin(distances))
                self.last_rejection={'kind':'contact_panel_collision','gap':float(distances[index]),'sphere':moving[index].tolist(),'point':panel[nearest[index]].tolist()};return False
        for box in exact_panel_boxes:
            from .compiled_collision import point_box_minimum
            if len(panel):
                gap=float(point_box_minimum(panel,base_rotation@box.center+offset,base_rotation@box.axes,box.half))
                if gap<margin:
                    self.last_rejection={'kind':'contact_panel_base_box','gap':gap,'part':box.name};return False
        return True


    def clear_contact_push(self, q, offset, margin=.005, base_yaw=0.):
        """Predict sensed leaf motion from a coordinated base/arm push pose."""
        co,si=np.cos(base_yaw),np.sin(base_yaw);base_rotation=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
        tcp=base_rotation@(self.geometry.fk(q)[self.geometry.tcp_body]@self.geometry.tcp_local)[:3,3]+offset
        a=self.reference_tcp[:2]-self.hinge_xy;b=tcp[:2]-self.hinge_xy
        angle=np.arctan2(a[0]*b[1]-a[1]*b[0],a@b)
        co,si=np.cos(angle),np.sin(angle);rotation=np.array([[co,-si,0],[si,co,0],[0,0,1.]])
        center=np.r_[self.hinge_xy,0.]
        return self.clear_contact_retreat(q,offset,margin,base_yaw=base_yaw,panel_motion=(rotation,center))
