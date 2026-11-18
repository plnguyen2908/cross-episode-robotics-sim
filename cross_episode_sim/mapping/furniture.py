"""Furniture landmarks from onboard pointing, masks, depth and estimated poses."""
import copy
import numpy as np
from PIL import Image

LABELS=('dining table','kitchen')
# Ground the work surface, not the containing room.
POINTING_LABELS={'dining table':'dining table','kitchen':'kitchen counter'}

class FurnitureMap:
    def __init__(self,pointer,segmenter,remembered=()):
        self.pointer=pointer;self.segmenter=segmenter;self.landmarks=[];self.events=[]
        # This task has one kitchen station and one dining-table destination.
        # Collapse legacy surface fragments; retain the strongest observed anchor.
        for label in LABELS:
            candidates=[x for x in remembered if x['label']==label or (label=='kitchen' and x['label']=='kitchen counter')]
            if candidates:
                item=copy.deepcopy(max(candidates,key=lambda x:x['observations']))
                item['label']=label;self.landmarks.append(item)
        self.query_views={label:[] for label in LABELS}

    def add(self,label,points,pose,timestamp):
        points=np.asarray(points)
        if label not in LABELS or len(points)<30 or not np.isfinite(points).all():return
        lo,hi=np.percentile(points,[5,95],axis=0);position=np.median(points,axis=0)
        matches=[x for x in self.landmarks if x['label']==label]
        if matches:
            item=min(matches,key=lambda x:np.linalg.norm(np.array(x['position'])-position))
            if timestamp<=item['last_seen']:return item
        else:
            item=dict(id=f'furniture_{len(self.landmarks):03d}',label=label,observations=0,viewpoints=[],evidence=[])
            self.landmarks.append(item)
        item.update(position=position.tolist(),observed_bounds=[lo.tolist(),hi.tolist()],last_seen=float(timestamp),observations=item['observations']+1)
        item['evidence'].append(dict(timestamp=float(timestamp),points=len(points)))
        # A sensor-observed robot viewpoint, not a simulator furniture parking pose.
        # A stand-off the robot already reached is the station; later remote sightings do not add to it.
        if not item.get('reached') and not any(np.linalg.norm(np.array(p)-pose[:3,3])<.2 for p in item['viewpoints']):item['viewpoints'].append(pose[:3,3].tolist())
        return item

    def queries(self, pose, timestamp):
        """Only novel views or bounded confirmation of a positive observation.

        Cache negatives too. Time alone must not restart queries at an unchanged
        viewpoint. Different rooms/views remain eligible even for known labels.
        View change alone paces discovery: a scan sweeps one field of view in
        about two seconds, so a timer between new views skips most of a turn.
        """
        xy=np.asarray(pose[:2,3]);yaw=np.arctan2(pose[1,0],pose[0,0])
        due=[]
        for label in LABELS:
            if any(x['label']==label and x['observations']>=2 for x in self.landmarks):continue
            nearby=[]
            for view in self.query_views[label]:
                angle=abs(np.arctan2(np.sin(yaw-view['yaw']),np.cos(yaw-view['yaw'])))
                if np.linalg.norm(xy-view['xy'])<=.75 and angle<=np.pi/4:
                    nearby.append(view)
            if nearby:
                view=min(nearby,key=lambda v:np.linalg.norm(xy-v['xy']))
                if not view['pending'] or view['attempts']>=3 or timestamp-view['last']<.8:
                    continue
                reason='confirm_candidate'
            else:
                view=dict(xy=xy.copy(),yaw=yaw,pending=False,attempts=0,last=-float('inf'))
                self.query_views[label].append(view)
                reason='new_view'
            due.append((label,view,reason))
        return due

    def observe(self,packet,tracking):
        if not tracking.accepted:return
        due=self.queries(tracking.T_map_base,packet.timestamp)
        if not due:return
        camera=packet.cameras['navigation']
        _,masks=self.segmenter.segment(camera.rgb[:,:,::-1].copy());masks=[np.asarray(m,bool) for m in masks]
        for label,view,reason in due:
            result=self.pointer.point(Image.fromarray(camera.rgb),POINTING_LABELS[label])
            used=[];observed=[]
            for x,y in result['points']:
                choices=[m for m in masks if m.shape==camera.depth_m.shape and m[int(y),int(x)] and m.sum()>=100]
                if not choices:continue
                mask=min(choices,key=lambda m:m.sum())
                if any(np.array_equal(mask,m) for m in used):continue
                used.append(mask)
                valid=mask & camera.valid_depth & np.isfinite(camera.depth_m) & (camera.depth_m>.1) & (camera.depth_m<6.)
                if camera.robot_mask is not None:valid &= ~camera.robot_mask
                from cross_episode_sim.perception.surfaces import table_surface_mask,project_mask
                T=tracking.T_map_base@camera.T_base_optical
                valid=table_surface_mask(camera.depth_m,valid,camera.K,T)
                points=project_mask(camera.depth_m,valid,camera.K,T)
                item=self.add(label,points,tracking.T_map_base,packet.timestamp)
                if item is not None:observed.append(item)
            view['attempts']+=1;view['last']=packet.timestamp
            # A missed confirmation may retry, but never indefinitely.
            view['pending']=(any(x['observations']<2 for x in observed) if observed else view['pending'])
            self.events.append(dict(timestamp=packet.timestamp,label=label,reason=reason,attempt=view['attempts'],raw=result['raw'],inference_seconds=result['wall_seconds']))


def furniture_viewpoints(memory,label):
    if label not in LABELS:raise ValueError(f'Unsupported furniture category: {label}')
    return [np.asarray(p)[:2] for item in memory.get('furniture',[]) if item['label']==label and item['observations']>=2 for p in item['viewpoints']]
