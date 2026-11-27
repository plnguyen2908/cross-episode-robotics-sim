"""Relocalize onboard RGB-D against a read-only, fine-tuning history memory.

No simulator, episode reset metadata, commands or world poses are consulted.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ..slam.keyframes import Keyframe, KeyframeTracker
from .history_routes import alignment_consensus


def load_keyframes(directory):
    with np.load(Path(directory)/'keyframes.npz') as saved:
        return [Keyframe(**{key: saved[f'{i}_{key}'] for key in
                           ('pixels','points','descriptors','pose','index')})
                for i in range(len(saved.files)//5)]


class HistoryRelocalizer:
    """Short RGB-D sequence -> initial pose in an existing history-map frame.

    Verification requires three query frames and two reference keyframes.
    Relative query motion is itself estimated by RGB-D registration, not read
    from commands or assumed stationary. This is startup localization, not a
    replacement for subsequent online tracking.
    """
    def __init__(self, memory_root, merged, feature_backend='cuda_sift'):
        self.root = Path(memory_root)
        self.merged = Path(merged)
        self.graph = json.loads((self.merged/'graph.json').read_text())
        if len(set(self.graph['components'].values())) != 1:
            raise ValueError('Select one registered memory component first')
        self.references = []
        for episode, meta in self.graph['episodes'].items():
            directory = self.root/'episodes'/episode
            calibration = json.loads((directory/'calibration.json').read_text())
            for key in load_keyframes(directory):
                key.pose = np.asarray(meta['transform']) @ key.pose
                self.references.append((episode, key, calibration))
        self.feature_backend = feature_backend
        # Use saved measured geometry to check image-match hypotheses. This is
        # the history reconstruction, never a simulator collision map.
        cfg=json.loads((self.merged/'component_0/summary.json').read_text())['config']
        with np.load(self.merged/'component_0/navigation_map.npz') as saved:
            rows,cols,heights=np.where(saved['evidence']>=2)
            points=np.column_stack((saved['origin_xy'][0]+(cols+.5)*float(saved['resolution']),
                                    saved['origin_xy'][1]+(rows+.5)*float(saved['resolution']),
                                    cfg['min_obstacle_height']+(heights+.5)*cfg['height_resolution']))
        self.geometry=cKDTree(points)

    def geometry_fit(self, camera, pose):
        from ..perception.surfaces import project_mask
        valid=camera.valid_depth & np.isfinite(camera.depth_m) & (camera.depth_m>.65) & (camera.depth_m<6.)
        if camera.robot_mask is not None:valid &= ~camera.robot_mask
        points=project_mask(camera.depth_m,valid,camera.K,camera.T_base_optical)
        points=points[(points[:,2]>.2)&(points[:,2]<2.5)]
        if len(points)>3000:points=points[np.linspace(0,len(points)-1,3000,dtype=int)]
        if len(points)<100:return dict(accepted=False,reason='insufficient structural depth')
        # A uniform wall/floor patch does not identify a unique room even when
        # repeated texture gives many image correspondences.
        spread=np.linalg.svd(points-np.mean(points,axis=0),compute_uv=False)/np.sqrt(len(points))
        mapped=points@pose[:3,:3].T+pose[:3,3]
        distances=self.geometry.query(mapped)[0]
        fraction=float(np.mean(distances<.15));median=float(np.median(distances))
        return dict(accepted=bool(fraction>=.7 and median<.08 and spread[-1]>.05),
                    fraction_within_15cm=fraction,median_distance_m=median,
                    minimum_structural_spread_m=float(spread[-1]),points=len(points))

    def localize(self, cameras, frame_ids=None, *, relative_poses=None, min_view_angle_degrees=0., retrieval=None, room_recovery=True):
        """Optionally supply RGB-D-tracked scan poses relative to scan start.

        These must come from observation-based tracking, never reset metadata
        or commanded yaw integration. Supplying them allows views after large
        turns that no longer overlap the very first image.
        """
        if len(cameras) < 3:
            raise ValueError('Provide at least three RGB-D frames')
        start = time.monotonic()
        frame_ids = list(range(len(cameras))) if frame_ids is None else list(frame_ids)
        if len(frame_ids)!=len(cameras) or len(set(frame_ids))!=len(frame_ids):
            raise ValueError('Provide one unique frame ID for each query camera')
        if relative_poses is not None:
            relative_poses=np.asarray(relative_poses)
            if relative_poses.shape!=(len(cameras),4,4) or not np.isfinite(relative_poses).all():
                raise ValueError('One finite estimated relative pose per query view is required')
        first = cameras[0]
        mount = first.T_base_optical
        tracker = KeyframeTracker(first.K, mount, feature_backend=self.feature_backend)
        for _,_,cal in self.references:
            if not np.allclose(cal['K'],first.K) or not np.allclose(cal['mount'],mount):
                raise ValueError('Reference/query camera calibration differs; calibrate before matching')
        if retrieval is not None:
            from .hierarchical_history import reference_signature
            if retrieval.signature != reference_signature(self.references):
                raise ValueError('Retrieval index does not belong to this history memory')
        matches=[];features=[];local_poses=[];anchor=None;retrieval_trace=[];registration_attempts=0
        for query_index,(frame_id, c) in enumerate(zip(frame_ids, cameras)):
            if not np.allclose(first.K,c.K) or not np.allclose(mount,c.T_base_optical):
                raise ValueError('Changing camera calibration in query sequence')
            depth = c.depth_m.copy()
            valid = c.valid_depth & np.isfinite(depth) & (depth >= .65) & (depth <= 6.)
            if c.robot_mask is not None:
                valid &= ~c.robot_mask
            depth[~valid] = 0
            uv,xyz,desc = tracker.features(c.rgb,depth)
            features.append((uv,xyz,desc))
            if relative_poses is not None:
                local=relative_poses[query_index]
            elif anchor is None:
                anchor=Keyframe(uv,xyz,desc,mount.copy(),frame_id)
                local=np.eye(4)
            else:
                relative=tracker.register(anchor,uv,xyz,desc)
                if relative is None:
                    local_poses.append(None)
                    continue
                local=relative[0] @ np.linalg.inv(mount)
            local_poses.append(local)
            if retrieval is None:
                candidates=range(len(self.references))
            else:
                candidates,trace=retrieval.select(c.rgb)
                trace['source_frame']=int(frame_id);trace['local_results']=[]
                retrieval_trace.append(trace)
            for reference_index in candidates:
                episode,ref,_=self.references[reference_index]
                registration_attempts+=1
                result=tracker.register(ref,uv,xyz,desc)
                if retrieval is not None:
                    trace['local_results'].append(dict(reference_index=reference_index,
                        registered=result is not None,inliers=int(result[1]) if result else 0))
                if result is None:
                    continue
                optical_pose,n,rmse=result
                base_pose=optical_pose @ np.linalg.inv(mount)
                matches.append(dict(transform=(base_pose @ np.linalg.inv(local)).tolist(),
                                    query_pose=base_pose.tolist(),source_frame=int(frame_id),
                                    reference_frame=f'{episode}:{int(ref.index)}',reference_index=reference_index,
                                    reference_episode=episode,reference_image_index=int(ref.index),inliers=n,rmse=rmse))
        query_by_id=dict(zip(frame_ids,cameras))
        local_by_id=dict(zip(frame_ids,local_poses))
        for match in matches:
            # Each rotated view provides its own independent structural
            # evidence; a blank first view must not veto every later view.
            view=query_by_id[match['source_frame']] if relative_poses is not None else first
            pose_key='query_pose' if relative_poses is not None else 'transform'
            match['geometry_fit']=self.geometry_fit(view,np.array(match[pose_key]))
        eligible=[i for i,m in enumerate(matches) if m['geometry_fit']['accepted']]
        pose,ids=alignment_consensus([matches[i] for i in eligible])
        consensus=[eligible[i] for i in ids]
        # A second equally supported spatial explanation is an ambiguity, not
        # evidence that the largest-inlier match identifies the correct room.
        other=[m for i,m in enumerate(matches) if i in eligible and i not in consensus]
        alternate,alternate_ids=alignment_consensus(other)
        ambiguous=(pose is not None and alternate is not None and len(alternate_ids)>=.8*len(consensus))
        supported_ids=sorted({matches[i]['source_frame'] for i in consensus})
        headings=np.array([np.arctan2(local_by_id[i][1,0],local_by_id[i][0,0]) for i in supported_ids])
        span=0.
        for angle in headings:
            if len(headings):span=max(span,float(np.max(np.abs(np.arctan2(np.sin(headings-angle),np.cos(headings-angle))))))
        diverse=bool(np.rad2deg(span)>=min_view_angle_degrees)
        accepted=bool(pose is not None and not ambiguous and diverse)
        best=max(consensus,key=lambda i:matches[i]['inliers']) if consensus else None
        report=dict(accepted=accepted,status='localized' if accepted else ('ambiguous' if ambiguous else 'insufficient_view_diversity' if pose is not None and not diverse else 'no_consistent_match'),
                    T_map_initial_base=pose.tolist() if accepted else None,
                    query_frames=frame_ids,reference_keyframes=len(self.references),matches=matches,
                    consensus_indices=consensus,alternate_consensus_size=len(alternate_ids),best_match_index=best,
                    wall_seconds=time.monotonic()-start,
                    relative_motion_source='provided RGB-D tracking' if relative_poses is not None else 'direct RGB-D registration to first view',
                    supported_view_angle_degrees=float(np.rad2deg(span)),required_view_angle_degrees=float(min_view_angle_degrees),
                    feature_policy='Image matches propose poses; non-floor metric depth must agree with saved history geometry and show non-planar structure',
                    scope='Offline startup relocalization only; no test poses/actions/room labels used; memory is read-only')
        report['retrieval']=dict(mode='hierarchical_netvlad_rgbd' if retrieval is not None else 'exhaustive',
            trace=retrieval_trace,registration_attempts=registration_attempts,
            exhaustive_pair_budget=len(cameras)*len(self.references))
        report['localization_method']='image_matching'
        report['room_recovery_used']=False
        if not accepted and room_recovery and relative_poses is not None:
            from .room_geometry_localization import RoomGeometryLocalizer, reconstruct_history, reconstruct_scan
            recovery_start=time.monotonic()
            if not hasattr(self,'_room_geometry'):
                self._room_geometry=RoomGeometryLocalizer(reconstruct_history(self))
                history_prepare_seconds=time.monotonic()-recovery_start
            else:
                history_prepare_seconds=0.
            scan_start=time.monotonic()
            self._last_room_scan=reconstruct_scan(cameras,relative_poses)
            reconstruction_seconds=time.monotonic()-scan_start
            geometry=self._room_geometry.localize(self._last_room_scan)
            geometry.update(history_prepare_seconds=history_prepare_seconds,
                            reconstruction_seconds=reconstruction_seconds,
                            recovery_total_seconds=time.monotonic()-recovery_start)
            image_report=report
            accepted=geometry['accepted']
            pose=np.asarray(geometry['T_map_initial_base']) if accepted else None
            report=dict(image_report,accepted=accepted,status=geometry['status'],
                        T_map_initial_base=pose.tolist() if accepted else None,
                        localization_method='room_geometry',room_recovery_used=True,
                        image_result=image_report,room_recovery=geometry,
                        wall_seconds=time.monotonic()-start)
        if accepted:
            if relative_poses is not None:
                report['T_map_current_base']=(pose@relative_poses[-1]).tolist()
            node=min(self.graph['nodes'],key=lambda x:np.linalg.norm(np.asarray(x['pose'])[:2,3]-pose[:2,3]))
            report['nearest_route_node']=dict(id=node['id'],distance_m=float(np.linalg.norm(np.asarray(node['pose'])[:2,3]-pose[:2,3])),
                                             connection_verified=False)
            landmarks=[]
            for filename in ('semantic_memory.json','receptacle_memory.json'):
                path=self.merged/filename
                if not path.exists():continue
                for x in json.loads(path.read_text())['furniture']:
                    if filename=='receptacle_memory.json' and not x.get('confirmed'):continue
                    landmarks.append(dict(label=x['label'],distance_m=float(np.linalg.norm(np.asarray(x['position'])[:2]-pose[:2,3]))))
            report['nearby_landmarks']=sorted(landmarks,key=lambda x:x['distance_m'])
            report['room_assignment']='Not implemented; landmark proximity is not a room polygon'
        return report,features
