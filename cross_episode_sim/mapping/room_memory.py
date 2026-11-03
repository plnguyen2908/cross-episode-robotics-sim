"""Visual room labels attached to sensor-derived Hydra place regions."""
import json
from collections import Counter
import numpy as np
from PIL import Image

ROOM_LABELS={'kitchen','dining room','living room','bedroom','bathroom','hallway','unknown'}
ROOM_PROMPT='''Classify the room surrounding this robot camera using visible furniture and appliances. Do not confuse a room seen through a doorway with the room containing the camera. If uncertain say unknown. Return JSON only with label (kitchen, dining room, living room, bedroom, bathroom, hallway, unknown), confidence (0 to 1), and evidence (short visible cues).'''

class ObservedRoomMemory:
    def __init__(self,vlm,period=5.,remembered=()):
        self.vlm=vlm;self.period=period;self.last_query=-float('inf')
        self.regions={};self.votes={};self.context=[];self.current_label='unknown';self.events=[];self.current_region=None
        self.remembered=list(remembered)
    def _region_at(self,xy):
        choices=[]
        for key,region in self.regions.items():
            places=np.asarray(region.get('places',[]),float)
            if places.ndim!=2 or not len(places):continue
            distance=float(np.linalg.norm(places[:,:2]-xy,axis=1).min())
            if distance<.8:choices.append((distance,key))
        choices.sort()
        if len(choices)>1 and choices[1][0]-choices[0][0]<.2:return None
        return choices[0][1] if choices else None
    def update(self,packet,tracking,regions):
        if not tracking.accepted:return
        self.regions={r['id']:r for r in regions}
        region=self._region_at(tracking.T_map_base[:2,3])
        if region!=self.current_region:
            self.context=[];self.current_label='unknown'
        self.current_region=region
        # Confirm an initial visual vote before a scan rotates to another room.
        # Established labels retain the normal rate; two independent frames are
        # still required and neither room IDs nor labels come from scene truth.
        pending=len(self.context)==1 and self.context[0]!='unknown'
        interval=min(self.period,.5) if pending else self.period
        if packet.timestamp-self.last_query<interval:return
        self.last_query=packet.timestamp
        raw=self.vlm.label_with_prompt([Image.fromarray(packet.cameras['navigation'].rgb)],ROOM_PROMPT)
        try:
            text=raw.strip()
            if text.startswith('```'):text=text.split('\n',1)[-1].rsplit('```',1)[0].strip()
            value=json.loads(text);label=value['label'];confidence=float(value['confidence'])
            if label not in ROOM_LABELS or not np.isfinite(confidence) or not 0<=confidence<=1:raise ValueError('Invalid room label')
        except (ValueError,KeyError,TypeError):value={'label':'unknown','confidence':0.,'evidence':'Invalid model response'};label='unknown';confidence=0.
        accepted=label if confidence>=.8 else 'unknown'
        self.context=(self.context+[accepted])[-2:]
        self.current_label=accepted if self.context==[accepted,accepted] else 'unknown'
        if self.current_region is not None:
            votes=self.votes.setdefault(self.current_region,[]);votes.append(accepted);del votes[:-3]
        self.events.append(dict(timestamp=packet.timestamp,region=self.current_region,estimated_position=tracking.T_map_base[:3,3].tolist(),confirmed_label=self.current_label,**value))
    def label_for(self,key):
        votes=self.votes.get(key,[])
        if len(votes)<2 or votes[-1]=='unknown':return 'unknown'
        return votes[-1] if Counter(votes)[votes[-1]]>=2 else 'unknown'
    def destinations(self,label,xy):
        result=[np.asarray(r['position'][:2],float) for r in self.remembered if r['label']==label]
        for key,r in self.regions.items():
            if self.label_for(key)!=label:continue
            for point in r.get('places',[]):result.append(np.asarray(point[:2],float))
        return sorted(result,key=lambda p:np.linalg.norm(p-xy))
