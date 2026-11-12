"""Locate a visible target for base approach; never certify manipulation."""
import numpy as np
from PIL import Image


def locate_approach_target(packet, pointer, target):
    from ..control.sensor_appliance_backend import camera_points, pixel_point
    for name in ('navigation', 'wrist'):
        camera=packet.cameras[name]
        _,_,valid=camera_points(camera)
        result=pointer.point(Image.fromarray(camera.rgb), target)
        for pixel in result['points']:
            try:point=pixel_point(camera,valid,pixel)
            except ValueError:continue
            if np.isfinite(point).all():
                return point,dict(camera=name,pixel=list(pixel),target=target,
                    approach_anchor_only=True,release_authorized=False)
    raise ValueError('No visible RGB-D target for approach: '+target)
