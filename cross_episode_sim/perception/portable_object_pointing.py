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
    return PointingSupervisor(pointer, segmenter, label)


class GroceryPointer:
    """Clarify visible packaging without object IDs, colors or scene bindings."""
    descriptions = {
        'packaged drink': 'drink bottles, canned beverages, or clearly labeled beverage cartons, excluding plain rectangular snack boxes',
        'packaged snack': 'rectangular snack boxes, excluding drink bottles and cups',
    }

    def __init__(self, pointer, category):
        self.pointer = pointer
        self.description = self.descriptions[category]

    def point(self, image, target):
        return self.pointer.point(image, self.description)
