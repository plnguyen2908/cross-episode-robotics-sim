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


# A countertop microwave is about 0.45 m tall and 0.75 m wide: its door handle is
# within this height band and reach of any point on its body. A cabinet-door
# handle under the counter (0.43 m below the microwave's, same x, in fix102) is not.
APPLIANCE_BODY_HEIGHT_BAND=.30
APPLIANCE_BODY_REACH=.60


def appliance_body_points(pointer,camera,valid,appliance):
    """Observed 3D points (camera_points frame) where the pointer marks the appliance itself."""
    import numpy as np
    from PIL import Image
    from ..control.sensor_appliance_backend import pixel_point
    bodies=[]
    for pixel in pointer.point(Image.fromarray(camera.rgb),appliance).get('points',[]):
        try:local=pixel_point(camera,valid,pixel)
        except ValueError:continue
        if np.isfinite(local).all():bodies.append(np.asarray(local,float))
    return bodies


def on_appliance_body(point,bodies):
    """True when a handle point lies in the height band and reach of a pointed appliance body."""
    import numpy as np
    point=np.asarray(point,float)
    return any(abs(point[2]-b[2])<=APPLIANCE_BODY_HEIGHT_BAND and np.linalg.norm(point[:2]-b[:2])<=APPLIANCE_BODY_REACH
               for b in bodies)
