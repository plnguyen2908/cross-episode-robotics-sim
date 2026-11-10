"""Fixture-specific semantic association before accepting a handle point.

A visually plausible handle is not necessarily on the requested appliance.
Only onboard RGB is used; no fixture IDs or positions enter the verifier.
"""
from PIL import ImageDraw
from .molmo_pointing import PointingSupervisor
from .daaam_grounding import LocalVLM,parse_label


class ApplianceAssociatedPointer:
    def __init__(self,pointer,appliance='microwave'):
        self.pointer=pointer;self.appliance=appliance;self.verifier=LocalVLM(device='cuda:0')
        self.events=[]

    def point(self,image,target):
        result=self.pointer.point(image,target)
        accepted=[]
        for point in result.get('points',[]):
            marked=image.copy();draw=ImageDraw.Draw(marked);x,y=point
            draw.ellipse((x-7,y-7,x+7,y+7),outline='magenta',width=3)
            answer=self.verifier.label_with_prompt([image,marked],
                'The second image marks one proposed contact point with a magenta circle. '
                'Which appliance does that point belong to? '
                'Distinguish countertop microwaves from ovens; hinged cabinet doors from horizontal drawers; and refrigerators from storage cupboards. '
                'Return JSON category "'+self.appliance+'" if it belongs to that appliance (including door, frame or control panel), '
                'otherwise "other" or "unknown", and a short description.',max_new_tokens=120)
            try:valid=parse_label(answer,(self.appliance,))['category']==self.appliance
            except (ValueError,KeyError):valid=False
            self.events.append(dict(point=list(point),answer=answer,accepted=valid))
            if valid:accepted.append(point)
        return dict(result,points=accepted)


def fixture_pointing_supervisor(pointer,segmenter,label):
    if any(word in label.lower() for word in ('microwave','cabinet','drawer','fridge')):
        pointer=ApplianceAssociatedPointer(pointer,label.removeprefix('handle on '))
    return PointingSupervisor(pointer,segmenter,label)
