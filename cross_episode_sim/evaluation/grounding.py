"""Oracle-perception baseline: persistent visible IDs, masks and colored overlays.

Simulator body/role mappings remain private. Identity is learned on first visibility,
retained through occlusion, and shared by oracle and VLA execution adapters.
"""
import colorsys
from copy import deepcopy

import numpy as np
from PIL import Image, ImageDraw


class TargetRegistry:
    def __init__(self, model, bindings):
        self._roles = {b['role']: b['body'] for b in bindings}
        self._geom_roles = {}
        roots = {model.body(body).id: role for role, body in self._roles.items()}
        for gid, body in enumerate(model.geom_bodyid):
            body = int(body)
            while body and body not in roots:
                body = int(model.body_parentid[body])
            if body in roots:
                self._geom_roles[gid] = roots[body]
        self._ids = {}
        self._known = {}
        self.targets = []
        self.masks = {}

    def update(self, images, segmentations):
        import mujoco
        visible = {}
        masks = {}
        for camera, segmentation in segmentations.items():
            ids = segmentation[:, :, 0]
            is_geom = segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            for gid in np.unique(ids[is_geom]):
                role = self._geom_roles.get(int(gid))
                if role is not None:
                    key = (camera, role)
                    mask = (ids == gid) & is_geom
                    masks[key] = masks.get(key, False) | mask
        for (camera, role), mask in masks.items():
            if int(mask.sum()) < 4:
                continue
            if role not in self._ids:
                number = len(self._ids) + 1
                identifier = f'obj_{number}'
                color = [round(c * 255) for c in colorsys.hsv_to_rgb(((number-1)*.61803398875) % 1, .8, 1)]
                self._ids[role] = identifier
                self._known[identifier] = dict(target_id=identifier, color_rgb=color,
                    category=role.split('_')[0] if role.startswith(('cup', 'bowl')) else 'object')
            identifier = self._ids[role]
            ys, xs = np.nonzero(mask)
            visible.setdefault(identifier, {})[camera] = dict(
                bbox_xyxy=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                mask_key=f'{camera}:{identifier}')
        self.targets = [dict(item, visible=identifier in visible, views=visible.get(identifier, {}))
                        for identifier, item in self._known.items()]
        self.masks = {f'{camera}:{self._ids[role]}': mask for (camera, role), mask in masks.items()
                      if role in self._ids and camera in visible.get(self._ids[role], {})}
        annotated = {}
        for camera, array in images.items():
            image = Image.fromarray(array).convert('RGB')
            occupied = []
            for target in self.targets:
                if camera not in target['views']:
                    continue
                color = tuple(target['color_rgb'])
                view = target['views'][camera]
                mask = self.masks[view['mask_key']]
                overlay = np.asarray(image).copy()
                overlay[mask] = (.75*overlay[mask] + .25*np.asarray(color)).astype(np.uint8)
                image = Image.fromarray(overlay)
                draw = ImageDraw.Draw(image)
                x0, y0, x1, y1 = view['bbox_xyxy']
                draw.rectangle((x0, y0, x1-1, y1-1), outline=color, width=2)
                label = target['target_id']
                bounds = draw.textbbox((0, 0), label)
                width, height = bounds[2]+4, bounds[3]+2
                x = max(0, min(x0, image.width-width))
                candidates = [max(0, y0-height), min(image.height-height, y1+2),
                              *range(0, max(1, image.height-height), height+2)]
                for y in candidates:
                    box = (x, y, x+width, y+height)
                    if not any(box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1]
                               for b in occupied):
                        break
                occupied.append(box)
                draw.line(((x0+x1)//2, (y0+y1)//2, x+width//2, y+height//2), fill=color)
                draw.rectangle(box, fill=color)
                draw.text((x+2, y), label, fill='black')
            annotated[camera] = np.asarray(image).copy()
        return dict(images=annotated, raw_images=images, targets=deepcopy(self.targets),
                    target_masks={k: v.copy() for k, v in self.masks.items()},
                    perception='oracle_instance_segmentation')

    def catalog(self, catalog):
        result = deepcopy(catalog)
        for skill in result:
            schema = skill['parameters']
            if 'role' in schema['properties']:
                schema['properties'].pop('role')
                schema['properties']['target_id'] = dict(type='string',
                    description='Persistent obj_N ID from observed targets; never guess an unseen ID.')
                schema['required'] = ['target_id' if x == 'role' else x for x in schema.get('required', [])]
        return result

    def resolve(self, name, arguments, interface):
        arguments = deepcopy(arguments)
        if 'role' in arguments:
            raise ValueError('Use grounded target_id, not private role')
        if 'target_id' in arguments:
            identifier = arguments.pop('target_id')
            role = next((role for role, value in self._ids.items() if value == identifier), None)
            if role is None:
                raise ValueError('Target has never been observed')
            arguments['role'] = role
        interface.validate(name, arguments)
        return arguments


def grounded_observation(registry, renderer, controller, cameras, images):
    segmentation = {}
    try:
        renderer.enable_segmentation_rendering()
        for camera in cameras:
            renderer.update_scene(controller.data, camera=camera,
                                  scene_option=getattr(controller, 'render_scene_option', None))
            segmentation[camera] = renderer.render().copy()
    finally:
        renderer.disable_segmentation_rendering()
    return registry.update(images, segmentation)
