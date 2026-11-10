"""Closed door hypotheses from onboard RGB pointing and measured depth."""
import json
import numpy as np
from PIL import Image, ImageDraw
from cross_episode_sim.navigation.room_exploration_policy import PortalEvidence


class ObservedPortalMap:
    def __init__(self,pointer,verifier=None):
        self.pointer=pointer;self.verifier=verifier;self.tracks=[];self.events=[]

    def confirm(self, camera, pixel):
        """Pointing proposes a handle; it does not establish a closed door."""
        image=Image.fromarray(camera.rgb).copy();draw=ImageDraw.Draw(image)
        x,y=pixel;draw.ellipse((x-7,y-7,x+7,y+7),outline='red',width=2)
        prompt=('The red circle marks a proposed door handle. Classify the door belonging to this handle, '
                'using the full image. Is it an interior passage door that is closed across its doorway, '
                'an interior passage door standing open, or a different object (cabinet, appliance, furniture, wall)? '
                'Do not infer closed merely because a handle is visible. Return only JSON with category '
                'equal to closed_room_door, open_room_door, other, or unknown, and a brief reason.')
        raw=self.verifier.label_with_prompt([image],prompt,max_new_tokens=120)
        try:
            cleaned=raw.strip()
            if cleaned.startswith('```'):cleaned=cleaned.split('\n',1)[-1].rsplit('```',1)[0]
            category=json.loads(cleaned)['category']
            if category not in {'closed_room_door','open_room_door','other','unknown'}:category='unknown'
        except (ValueError,KeyError,TypeError):category='unknown'
        return category,raw

    def traversed(self, source, destination, before, after):
        """An accepted SLAM path links room footprints without a visible handle."""
        if source==destination:return None
        existing=next((t for t in self.tracks if t.get('traversed') and
                       {t['source'],t['destination']}=={source,destination}),None)
        if existing is not None:return existing
        before=np.asarray(before);after=np.asarray(after)
        delta=after[:2]-before[:2];norm=np.linalg.norm(delta)
        track=dict(portal=f'portal_{len(self.tracks):03d}',position=(before+after)/2,
                   observations=2,source=source,destination=destination,closed=False,
                   traversed=True,association='accepted_slam_room_transition',
                   normal=(delta/max(norm,1e-9)).tolist(),last_seen=0.)
        self.tracks.append(track)
        return track

    def mark_open(self,track,opened_handle=None):
        track['closed']=False
        track['opened_by_robot']=True
        if opened_handle is not None:
            track['opened_handle']=np.asarray(opened_handle).tolist()
            for other in self.tracks:
                if other is not track and np.linalg.norm(other['position']-opened_handle)<.2:
                    other['duplicate_of']=track['portal']

    def observe(self,packet,tracking,current_room):
        if not tracking.accepted:return
        camera=packet.cameras['navigation']
        result=self.pointer.point(Image.fromarray(camera.rgb),'handles of closed interior room doors (not cabinets or appliances)')
        T=tracking.T_map_base@camera.T_base_optical
        for x,y in result['points']:
            u,v=int(x),int(y)
            if not(0<=u<camera.depth_m.shape[1] and 0<=v<camera.depth_m.shape[0]):continue
            if not camera.valid_depth[v,u] or (camera.robot_mask is not None and camera.robot_mask[v,u]):continue
            z=float(camera.depth_m[v,u])
            point=T[:3,:3]@np.array([(u-camera.K[0,2])*z/camera.K[0,0],(v-camera.K[1,2])*z/camera.K[1,1],z])+T[:3,3]
            if not .5<point[2]<1.6:continue
            if any(not t['closed'] and t.get('opened_handle') is not None
                   and np.linalg.norm(point-t['opened_handle'])<.2 for t in self.tracks):continue
            matches=[t for t in self.tracks if np.linalg.norm(t['position']-point)<.12]
            if matches:
                t=min(matches,key=lambda t:np.linalg.norm(t['position']-point))
                t['position']=(t['position']*t['observations']+point)/(t['observations']+1)
                t['observations']+=1;t['last_seen']=packet.timestamp
                if current_room is not None and t['source'] is None:t['source']=current_room
            else:
                t=dict(portal=f'portal_{len(self.tracks):03d}',position=point,observations=1,source=current_room,destination=None,closed=True,last_seen=packet.timestamp)
                self.tracks.append(t)
            if self.verifier is not None and t['observations']>=2 and 'visual_category' not in t:
                category,raw=self.confirm(camera,(x,y))
                t['visual_category']=category;t['visual_response']=raw
                if category=='open_room_door':t['closed']=False
                self.events.append(dict(time=packet.timestamp,portal=t['portal'],category=category,raw=raw))
        self.events.append(dict(time=packet.timestamp,room=current_room,raw=result['raw'],points=len(result['points'])))

    def evidence(self,current_room):
        return [PortalEvidence(t['portal'],t['source'] or current_room,t['destination'],t['closed'],t['destination'] is not None,t.get('reachable',True) and not t.get('failed',False))
                for t in self.tracks if t['observations']>=2 and not t.get('duplicate_of')
                and (self.verifier is None or t.get('traversed') or t.get('opened_by_robot')
                     or t.get('visual_category')=='closed_room_door')]
