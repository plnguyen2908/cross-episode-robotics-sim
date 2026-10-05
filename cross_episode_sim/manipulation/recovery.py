"""Local alternate-stance recovery using shared physical A* and base execution."""
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from cross_episode_sim.manipulation.grasp_qualification import GraspQualification
from cross_episode_sim.navigation.planners import plan_same_room_route

from cross_episode_sim.manipulation.edge_access import ThinEdgeAccessMixin

class RecoveringManipulation(ThinEdgeAccessMixin, GraspQualification):
    def navigation_penetration(self, data, carrying):
        if getattr(self, "edge_pushing", False):
            return ThinEdgeAccessMixin.navigation_penetration(self, data, carrying)
        robot, _, payload = self.contact_body_masks()
        contacts = data.contact
        b1 = self.model.geom_bodyid[contacts.geom1]
        b2 = self.model.geom_bodyid[contacts.geom2]
        r1, r2 = robot[b1], robot[b2]
        p1, p2 = payload[b1], payload[b2]
        selected = r1 != r2
        if carrying:
            selected |= (p1 != p2) & ~(r1 | r2)
            selected &= ~((r1 & p2) | (r2 & p1))
        selected &= ~(self._contact_floor_mask[b1] | self._contact_floor_mask[b2])
        return float(np.max(.001 - contacts.dist[selected], initial=0.))

    def plan_route(self, goal, carrying, face=None):
        start=self.base_pose().copy();final=np.array([*goal,start[2] if face is None else face])
        probe=mujoco.MjData(self.model);probe.qpos[:]=self.data.qpos
        addresses=[self.model.jnt_qposadr[self.model.joint('robot_0/'+n).id] for n in self.profile.base_joints]
        object_address=self.model.jnt_qposadr[self.model.joint(self.object_joint).id]
        initial_qpos=self.data.qpos.copy()
        object_position=initial_qpos[object_address:object_address+3].copy()
        object_rotation=Rotation.from_quat(initial_qpos[object_address+3:object_address+7],scalar_first=True)
        def clear(pose):
            probe.qpos[:]=initial_qpos
            probe.qpos[addresses]=pose
            if carrying:
                rotation=Rotation.from_euler('z',pose[2]-start[2])
                probe.qpos[object_address:object_address+3]=rotation.apply(object_position-np.array([*start[:2],0.]))+np.array([*pose[:2],0.])
                probe.qpos[object_address+3:object_address+7]=(rotation*object_rotation).as_quat(scalar_first=True)
            mujoco.mj_forward(self.model,probe)
            return self.navigation_penetration(probe,carrying)<=0 and self.robot_self_penetration(probe)<=.0005
        # Restrict to the source room's floor and the local vicinity of the object.
        floor=[]
        origin=self.initial_object_pose[:2,3].copy()
        for g in range(self.model.ngeom):
            name=self.model.body(self.model.geom_bodyid[g]).name or ''
            if name.startswith('floor_'):
                rot=self.data.geom_xmat[g].reshape(3,3)
                local=rot.T@(np.array([*start[:2],self.data.geom_xpos[g,2]])-self.data.geom_xpos[g])
                if np.all(np.abs(local[:2])<=self.model.geom_size[g,:2]+.01): floor.append(g)
        if not floor: raise RuntimeError('No source-room floor found for recovery')
        def on_floor(pose):
            if np.linalg.norm(np.asarray(pose[:2])-origin)>1.6:return False
            for g in floor:
                xyz=np.array([*pose[:2],self.data.geom_xpos[g,2]])
                local=self.data.geom_xmat[g].reshape(3,3).T@(xyz-self.data.geom_xpos[g])
                if np.all(np.abs(local[:2])<self.model.geom_size[g,:2]-.05):return True
            return False
        last=None
        for reverse in [.15,0.]:
            try:
                route,metrics=plan_same_room_route(start,final,on_floor,clear,reverse=reverse)
                self._pickup_pre_nav_undock = reverse
                self.record(**metrics);return route
            except RuntimeError as exc:last=exc
        raise last

    def _plan_qualified_grasp(self):
        return super().select_annotated_grasp()

    def _plan_with_grasp_fallback(self):
        if hasattr(self, 'planner_world_boxes'):
            positions = [float(self.data.joint('robot_0/' + n).qpos[0])
                         for n in self.planner.names]
            clearance = self.planner.world_clearance(positions, self.planner_world_boxes)
            if clearance < 0:
                # A* checks real geometry; cuRobo's conservative spheres can
                # still overlap here. No grasp goal fixes an invalid start.
                self.record(grasp_start_blocked=True, planner_start_clearance_m=clearance,
                            recovery='try another dock before sampling grasps')
                raise RuntimeError('No annotated grasp passed robot reachability and collision checks')
        try:
            return self._plan_qualified_grasp()
        except RuntimeError as exc:
            if str(exc) != 'No annotated grasp passed robot reachability and collision checks': raise
            if getattr(self, '_qualified_angles_only', False): raise
            if self.args.annotation_source != 'qualified_registry': raise
            from cross_episode_sim.manipulation.molmo_objects import OUTPUT
            path=OUTPUT/self.annotation_asset/'grasps.npz'
            if self.asset_metadata['source'] != 'molmo' or not path.is_file(): raise
            self.record(saved_grasp_subset_exhausted=True, fallback_library=str(path))
            self.local_annotations=np.load(path)['transforms'].copy()
            self.annotation_path=path
            self.args.annotation_source='droid'
            self.annotation_vertical_offsets = getattr(
                self, 'annotation_recovery_vertical_offsets', (0.,))
            self.active_family='any'
            self.physically_rejected_annotation_variants=set()
            return super().select_annotated_grasp()

    def grasp_recovery_stances(self):
        target=self.bread_pose()[:2,3].copy();original=self.base_pose().copy()
        radial=original[:2]-target;angle=np.arctan2(radial[1],radial[0])
        for distance,offset in [(.50,0.),(.42,.35),(.42,-.35),(.48,.75),(.48,-.75),(.42,1.2),(.42,-1.2),(.60,.35),(.60,-.35),(.65,.75),(.65,-.75),(.75,1.2),(.75,-1.2)]:
            xy=target+distance*np.array([np.cos(angle+offset),np.sin(angle+offset)])
            yaw=float(np.arctan2(target[1]-xy[1],target[0]-xy[0]))
            yield np.array([*xy,yaw])

    def select_annotated_grasp(self):
        if getattr(self, '_trial_storage_dock', None) is not None:
            # StoragePickupSkill owns redocking/rollback. This branch plans only
            # at its nominated dock, retaining the existing grasp fallbacks.
            return self._plan_with_grasp_fallback()
        selected=getattr(self,'_edge_selected',None)
        if selected is not None:
            self._edge_selected=None
            return selected
        history=self.report.setdefault('stance_recovery_attempts',[])
        # A raw-library fallback is local to a dock. A different base pose can
        # make a saved successful grasp reachable; do not keep only the raw
        # library's small, differently ranked candidate subset after moving.
        qualified = None
        if self.args.annotation_source == 'qualified_registry':
            qualified = (self.local_annotations.copy(), self.annotation_path,
                         self.annotation_vertical_offsets,
                         set(self.physically_rejected_annotation_variants))
        try:return self._plan_with_grasp_fallback()
        except RuntimeError as exc:
            if str(exc)!='No annotated grasp passed robot reachability and collision checks':raise
            history.append(dict(stance=self.base_pose().tolist(),error=str(exc)))
        if getattr(self,'_recovery_exhausted',False):raise RuntimeError('Saved grasp plans exhausted after stance recovery')
        self._recovery_exhausted=True
        self.tuck_for_navigation()
        # Different distance and bearing; actual body sweeps validate every route.
        for dock in self.grasp_recovery_stances():
            xy,yaw=np.asarray(dock[:2]),float(dock[2])
            try:
                self.navigate(xy,False,face=yaw)
                self.report['navigation_tested']=True
                if qualified is not None:
                    (self.local_annotations, self.annotation_path,
                     self.annotation_vertical_offsets, rejected) = qualified
                    self.physically_rejected_annotation_variants = set(rejected)
                    self.args.annotation_source = 'qualified_registry'
                    self.active_family = 'any'
                    self.record(retry_saved_grasps_at_new_dock=True)
                self.planner=self.make_planner();self.arm_aids=self.actuator_ids(self.planner.names)
                self.load_world();self.preplanned_moves={}
                result=self._plan_with_grasp_fallback()
                history.append(dict(stance=self.base_pose().tolist(),planned=True))
                return result
            except RuntimeError as exc:
                history.append(dict(candidate=[*xy.tolist(),yaw],error=str(exc)))
                self.record(stance_recovery_rejected=str(exc))
        raise RuntimeError('No saved grasp plan after bounded alternate navigation stances')

    def transport_payload(self):
        self.prepare_loaded_manipulation()
        # Keep the successfully held orientation: restoring the object's initial
        # yaw unnecessarily imposed difficult wrist rotations on rounded objects.
        actual=self.bread_pose()
        local=(self.bread_vertices()-actual[:3,3])@actual[:3,:3]
        held=self.placement_reference_pose()
        corners=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
        top=-np.inf
        for g in self.table_gids:
            pts=self.data.geom_xpos[g]+(self.model.geom_aabb[g,:3]+corners*self.model.geom_aabb[g,3:])@self.data.geom_xmat[g].reshape(3,3).T
            top=max(top,float(pts[:,2].max()))
        placements=[]
        for dx,dy in [(-.14,0.),(.14,0.),(-.10,.03),(.10,.03),(0.,.10)]:
            pose=held.copy();pose[:2,3]=self.initial_object_pose[:2,3]+[dx,dy]
            pose[2,3]=top-float((local@pose[:3,:3].T)[:,2].min())
            placements.append(pose)
        self.destination_pose=placements[0];self.placement_pose_options=placements[1:]
        self.record(placement_orientation_policy='retain held orientation',placement_candidates=len(placements))

    def redock_loaded_for_placement(self):
        history=self.report.setdefault('loaded_redock_attempts',[])
        self.tuck_loaded_for_navigation()
        target=self.destination_pose[:2,3].copy()
        if not hasattr(self,'_placement_docks'):
            delta=self.base_pose()[:2]-target;angle=np.arctan2(delta[1],delta[0])
            self._placement_docks=[]
            for distance,offset in [(.50,0.),(.42,.4),(.42,-.4),(.50,.75)]:
                xy=target+distance*np.array([np.cos(angle+offset),np.sin(angle+offset)])
                yaw=float(np.arctan2(target[1]-xy[1],target[0]-xy[0]))
                self._placement_docks.append((xy,yaw))
        while self._placement_docks:
            xy,yaw=self._placement_docks.pop(0)
            try:
                self.navigate(xy,True,face=yaw)
                self.report['navigation_tested']=True
                self.prepare_loaded_manipulation()
                history.append(dict(stance=self.base_pose().tolist(),reached=True))
                self.record(loaded_placement_redock=self.base_pose().tolist())
                return
            except RuntimeError as exc:
                history.append(dict(candidate=[*xy.tolist(),yaw],error=str(exc)))
                if 'drop' in str(exc).lower():raise
        raise RuntimeError('No collision-free loaded placement redock among remaining candidates')

    def mesh_contact_move(self, stage, pose, path=None):
        try:
            return super().mesh_contact_move(stage, pose, path=path)
        except RuntimeError as exc:
            if stage != 'lower object onto destination table' or 'collision' not in str(exc).lower():raise
            # The arm has already reached a planned approach above the table.
            # If the real object's whole footprint is supported below, release
            # from this small height rather than force an unnecessary descent.
            vertices=self.bread_vertices()
            corners=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
            boxes=[]
            for g in self.table_gids:
                points=self.data.geom_xpos[g]+(self.model.geom_aabb[g,:3]+corners*self.model.geom_aabb[g,3:])@self.data.geom_xmat[g].reshape(3,3).T
                boxes.append(points)
            table=np.concatenate(boxes);low,high=table.min(0),table.max(0)
            clearance=float(vertices[:,2].min()-high[2])
            inside=bool(np.all(vertices[:,:2].min(0)>low[:2]+.01) and np.all(vertices[:,:2].max(0)<high[:2]-.01))
            if not (inside and .0 <= clearance <= .09):raise
            self.record(release_from_planned_approach=True,actual_release_clearance_m=clearance,descent_rejection=str(exc))
            return
