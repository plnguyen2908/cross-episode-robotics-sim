"""Point at visible basket floor before choosing its instance/part mask."""
import numpy as np
from PIL import Image
from .visual_targets import Detection


def pointed_basket_floor(rgb, semantic_label, output_label, pointer, masks):
    height,width=rgb.shape[:2]
    answer=pointer.point(Image.fromarray(rgb),
        'visible empty interior bottom INSIDE the '+semantic_label+
        ', between its walls, where an object would rest. Not its rim, handle, '
        'outer wall, counter, or held object. If the bottom is hidden return no points')
    result=[]
    for u,v in answer.get('points',[]):
        u,v=round(u),round(v)
        if not (0<=u<width and 0<=v<height):continue
        choices=[np.asarray(mask,bool) for mask in masks if np.shape(mask)==(height,width)
                 and mask[v,u] and np.count_nonzero(mask)>=30]
        if not choices:continue
        from .container_masks import enclosing_container_mask
        mask=enclosing_container_mask(min(choices,key=np.count_nonzero),masks)
        yy,xx=np.where(mask)
        result.append(Detection(output_label,1.,(float(xx.min()),float(yy.min()),
            float(xx.max()+1),float(yy.max()+1)),mask,(u,v)))
    print('BASKET_FLOOR_POINT',semantic_label,len(result),answer,flush=True)
    return result
