"""Synchronized diagnostic panels; never a policy observation.

Consume the execution stack's current map/pose and the same sensor packet used
for manipulation. Do not substitute simulator poses or a final reconstructed map.
"""
import numpy as np
from PIL import Image, ImageDraw
from .map_video import MapVideoPanel


class PipelineVideoPanel:
    def __init__(self, size=640):
        self.size = size
        self.map = MapVideoPanel()

    def render(self, grid, tracking, packet, phase, selected_tcp_base=None):
        """Optional TCP is M2T2's selected pose in the current base frame."""
        size = self.size
        left = self.map.render(grid, tracking, size, packet.timestamp)
        right = Image.new('RGB', (size, size + 80), 'white')
        draw = ImageDraw.Draw(right)
        draw.text((8, 8), f'M2T2 manipulation | {phase}', fill='black')
        draw.text((8, 25), f'Same sensor timestamp: {packet.timestamp:.2f}s', fill='black')
        draw.text((8, 42), 'Selected pose axes: X red / Y green / Z blue', fill='black')
        draw.text((8, 59), f'SLAM: {tracking.status}', fill='black')
        for index, key in enumerate(('navigation', 'wrist')):
            camera = packet.cameras[key]
            view = Image.fromarray(camera.rgb.copy())
            if selected_tcp_base is not None:
                tcp = np.asarray(selected_tcp_base)
                optical = np.linalg.inv(camera.T_base_optical)
                origin = tcp[:3, 3]
                xyz = np.vstack([origin, *(origin + .08 * tcp[:3, k] for k in range(3))])
                xyz = xyz @ optical[:3, :3].T + optical[:3, 3]
                pixels = xyz @ camera.K.T
                pixels = pixels[:, :2] / np.maximum(xyz[:, 2:3], 1e-6)
                pen = ImageDraw.Draw(view)
                for k, color in enumerate(('red', 'lime', 'blue'), 1):
                    if xyz[0, 2] > .01 and xyz[k, 2] > .01:
                        pen.line([tuple(pixels[0]), tuple(pixels[k])], fill=color, width=3)
            view.thumbnail((size, size // 2))
            right.paste(view, ((size-view.width)//2, 80 + index*size//2))
        canvas = Image.new('RGB', (size*2, size+80), 'white')
        canvas.paste(left, (0, 0)); canvas.paste(right, (size, 0))
        return np.asarray(canvas)
