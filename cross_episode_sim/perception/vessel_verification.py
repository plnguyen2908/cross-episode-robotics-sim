"""Confirm pointed bowls and cups from their segmented masks before they become targets.

Molmo2 answers "bowls" with a cup standing beside a bowl, and "cups" with bowls.
The verifier sees the pointed mask highlighted in the scene and its isolated
crop, and names the object independently of what was requested. It uses onboard
RGB, measured depth and the estimated base pose only.
"""
from collections import Counter
import time
import numpy as np
from .daaam_grounding import LocalVLM,PROMPT,masked_views,parse_label
from .molmo_pointing import PointingSupervisor
from .visual_targets import Detection,locate_detection

VESSELS=('bowl','cup')
# One object keeps being re-asked until this many verdicts agree with none against.
SETTLED_VOTES=3
SAME_OBJECT_RADIUS=.06
# A clearly closer view is a better look: ask again instead of reusing far verdicts.
CLOSER_LOOK=.75


class VerifiedVesselSupervisor(PointingSupervisor):
    def __init__(self,pointer,segmenter,label,verifier=None,**kwargs):
        super().__init__(pointer,segmenter,label,**kwargs)
        self.verifier=verifier;self.verdicts=[]

    def classify(self,camera,mask):
        if self.verifier is None:self.verifier=LocalVLM(device='cuda:0')
        answer=self.verifier.label_with_prompt(masked_views(camera.rgb,mask),PROMPT,max_new_tokens=120)
        try:return parse_label(answer)['category'],answer
        except (ValueError,KeyError):return 'unknown',answer

    def verify_mask(self,camera,mask,tracking):
        started=time.monotonic()
        yy,xx=np.where(mask)
        box=(float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1))
        position=locate_detection(camera,Detection(self.label,1.,box,mask=mask),tracking.T_map_base)
        entry=distance=None
        if position is not None:
            distance=float(np.linalg.norm(position[:2]-tracking.T_map_base[:2,3]))
            near=[e for e in self.verdicts if np.linalg.norm(e['position']-position)<SAME_OBJECT_RADIUS]
            entry=min(near,key=lambda e:np.linalg.norm(e['position']-position)) if near else None
        if entry is not None and distance>CLOSER_LOOK*entry['closest']:
            (category,votes),=entry['votes'].most_common(1)
            if category!='unknown' and votes>=SETTLED_VOTES and votes==sum(entry['votes'].values()):
                return dict(requested=self.label,category=category,accepted=category==self.label,remembered=True,
                            position=position.tolist(),wall_seconds=time.monotonic()-started)
        category,answer=self.classify(camera,mask)
        if position is not None:
            if entry is None:
                entry=dict(position=np.asarray(position,float),votes=Counter(),closest=distance);self.verdicts.append(entry)
            entry['votes'][category]+=1;entry['closest']=min(entry['closest'],distance)
        return dict(requested=self.label,category=category,accepted=category==self.label,response=answer,
                    position=None if position is None else position.tolist(),wall_seconds=time.monotonic()-started)
