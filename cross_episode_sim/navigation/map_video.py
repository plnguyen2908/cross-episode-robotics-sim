"""Diagnostic online map panel. Never supplied to the policy."""
import numpy as np
from PIL import Image,ImageDraw

class MapVideoPanel:
    def __init__(self):self.path=[]
    def render(self,grid,tracking,size,timestamp):
        floor=grid.floor_views>=grid.config.minimum_floor_views
        occupied=(grid.evidence>=2).any(axis=2)
        pixels=np.full((*floor.shape,3),45,np.uint8)
        pixels[floor]=(205,215,220);pixels[occupied]=(235,110,70)
        im=Image.fromarray(pixels);d=ImageDraw.Draw(im)
        def cell(xy):
            c,r=grid.world_to_cell(np.asarray(xy));return int(c),int(r)
        if tracking.accepted:self.path.append(tracking.T_map_base[:2,3].copy())
        points=[cell(x) for x in self.path]
        if len(points)>1:d.line(points,fill=(60,180,255),width=2)
        if points:
            c,r=points[-1];d.ellipse((c-2,r-2,c+2,r+2),fill=(60,255,90))
            if tracking.accepted and tracking.T_map_base is not None:
                heading=tracking.T_map_base[:2,0];e=cell(self.path[-1]+heading*.7);d.line([(c,r),e],fill=(60,255,90),width=2)
        # Fixed 20 m window in the initial SLAM frame, independent of ground truth.
        c,r=cell([0,0]);h=int(10/grid.config.resolution)
        im=im.crop((c-h,r-h,c+h,r+h)).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        im=im.resize((size,size),Image.Resampling.NEAREST)
        canvas=Image.new('RGB',(size,size+80),'white');canvas.paste(im,(0,80));d=ImageDraw.Draw(canvas)
        d.text((8,8),f'Online RGB-D map | {timestamp:.1f}s',fill='black')
        d.text((8,25),'Gray: unknown | light: observed floor',fill='black')
        d.text((8,42),'Orange: obstacles | blue: estimated path',fill='black')
        d.text((8,59),'Green: estimated robot | diagnostic only',fill='black')
        return canvas
