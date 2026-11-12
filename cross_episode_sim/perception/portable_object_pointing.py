"""Disambiguate portable food containers from countertop appliances using RGB."""
from PIL import ImageDraw
from cross_episode_sim.perception.daaam_grounding import LocalVLM, parse_label
from cross_episode_sim.perception.molmo_pointing import PointingSupervisor


class PortableFoodPointer:
    def __init__(self, pointer, verifier=None):
        self.pointer = pointer
        self.verifier = verifier if verifier is not None else LocalVLM(device='cuda:0')
        self.events = []

    def point(self, image, target):
        result = self.pointer.point(image, 'portable food container or food storage box')
        accepted = []
        for point in result.get('points', []):
            x, y = point
            marked = image.copy()
            ImageDraw.Draw(marked).ellipse((x-7, y-7, x+7, y+7), outline='magenta', width=3)
            half=48
            crop=marked.crop((max(0,int(x)-half),max(0,int(y)-half),min(image.width,int(x)+half),min(image.height,int(y)+half)))
            crop=crop.resize((384,384))
            answer = self.verifier.label_with_prompt([image, marked, crop],
                'The first image is context; the second marks a proposal; the third magnifies that proposal. Identify the object under the magenta point. '
                'Is it a portable container holding food, such as a meal box or food-storage container? '
                'A toaster, microwave, kettle, stove, fridge, counter or cabinet is an appliance/furniture, '
                'not a portable food container. Return JSON category "food container", "other" or '
                '"unknown", and a short description. Do not assume the proposal is correct.',
                max_new_tokens=120)
            try:
                valid = parse_label(answer, ('food container',))['category'] == 'food container'
            except (ValueError, KeyError):
                valid = False
            self.events.append(dict(point=list(point), response=answer, accepted=valid))
            if valid:
                accepted.append(point)
        return dict(result, points=accepted)


def household_pointing_supervisor(pointer, segmenter, label):
    if label == 'food container' or label.startswith('food container '):
        pointer = PortableFoodPointer(pointer)
    elif label in ('packaged drink', 'packaged snack'):
        pointer = GroceryPointer(pointer, label)
        return GroceryPointingSupervisor(pointer, segmenter, label)
    return PointingSupervisor(pointer, segmenter, label)


class GroceryPointingSupervisor(PointingSupervisor):
    def refine_mask(self, camera, mask):
        from .grocery_surface import above_support_mask
        return above_support_mask(camera, mask)


class GroceryPointer:
    """Clarify visible packaging without object IDs, colors or scene bindings."""
    from .target_language import GROCERY_DESCRIPTIONS as descriptions

    def __init__(self, pointer, category, verifier=None):
        self.pointer = pointer
        self.category = category
        self.description = self.descriptions[category]
        self.verifier = verifier

    def point(self, image, target):
        import time
        started=time.monotonic()
        result = self.pointer.point(image, self.description)
        verified=self.verify(image,result)
        verified['wall_seconds']=time.monotonic()-started
        return verified

    def verify(self,image,result):
        """Confirm proposals, including a projected sensor track, from current RGB."""
        import math,time
        started=time.monotonic()
        accepted, evidence = [], []
        if result.get('points') and self.verifier is None:
            self.verifier = LocalVLM(device='cuda:0')
        for point in result.get('points', []):
            x,y=map(float,point)
            if not (math.isfinite(x) and math.isfinite(y) and 0<=x<image.width and 0<=y<image.height):
                continue
            marked=image.copy()
            ImageDraw.Draw(marked).ellipse((x-6,y-6,x+6,y+6),outline='magenta',width=2)
            half=max(24,round(min(image.size)*.10))
            crop=image.crop((max(0,int(x)-half),max(0,int(y)-half),
                             min(image.width,int(x)+half),min(image.height,int(y)+half))).resize((384,384))
            answer=self.verifier.label_with_prompt([marked,crop],
                'Classify the actual object at the magenta point, using the full scene and unmarked crop. '
                'The proposal can be wrong. Choose independently of what was requested. '
                'packaged drink: identifiable beverage bottle, can, or printed beverage carton, with visible packaging evidence. '
                'packaged snack: a small plain solid-colored rectangular block/box used as a snack prop in this scene. '
                'A plain block is NOT a drink. A chair, robot, table or floor is other. '
                'Use unknown if occluded or too small to distinguish. '
                'Return JSON with category (packaged drink, packaged snack, other, unknown) and a short description of visible evidence.',
                max_new_tokens=120)
            try:
                label=parse_label(answer,tuple(self.descriptions))
            except (ValueError,KeyError):
                label=dict(category='unknown',description='Invalid classifier response')
            keep=label['category']==self.category
            evidence.append(dict(point=[x,y],requested=self.category,response=answer,accepted=keep))
            if keep:accepted.append(point)
        return dict(result,points=accepted,category_verification=evidence,
                    pointing_wall_seconds=result.get('wall_seconds'),wall_seconds=time.monotonic()-started)
