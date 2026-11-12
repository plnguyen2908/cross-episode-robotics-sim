"""Identify a container, then point at its interior in a magnified RGB crop."""
from PIL import Image
from .visual_targets import Detection
from .point_masks import PointMaskDetector


def container_interior_detections(rgb, semantic_label, output_label):
    detector=PointMaskDetector(categories=(semantic_label,))
    objects=detector.detect(rgb)
    result=[];height,width=rgb.shape[:2]
    for obj in objects:
        x0,y0,x1,y1=map(int,obj.box)
        padding=max(5,round(.12*max(x1-x0,y1-y0)))
        left=max(0,x0-padding);top=max(0,y0-padding)
        right=min(width,x1+padding);bottom=min(height,y1+padding)
        crop=Image.fromarray(rgb[top:bottom,left:right])
        scale=640/max(crop.size)
        resized=crop.resize((round(crop.width*scale),round(crop.height*scale)))
        answer=detector.pointer.point(resized,
            'center of the visible bare interior floor INSIDE the '+semantic_label+
            '. Point between its walls, on the bottom where an object would rest; '
            'not the top rim, handle, outer wall, counter, or an object. If its floor is not visible return no points')
        for u,v in answer.get('points',[]):
            u=round(left+u/resized.width*crop.width)
            v=round(top+v/resized.height*crop.height)
            if not (0<=u<width and 0<=v<height):continue
            if obj.mask is not None and not obj.mask[v,u]:continue
            result.append(Detection(output_label,obj.score,obj.box,obj.mask,(u,v)))
    return result


def microwave_floor_detections(camera, output_label):
    """Point at the visible turntable; grow only measured local depth.

    The appliance body's instance mask can exclude the cavity (a hole in that
    mask). The semantic floor point and connected horizontal-depth selector
    therefore supply the interior region, without using an outer-body mask.
    """
    from .molmo_pointing import MolmoPointer
    from .point_region import pointed_region
    description=('center of the circular glass turntable on the bottom INSIDE '
                 'the open microwave cavity, not the kitchen counter, '
                 'not the open door, and not the container held by the robot')
    answer=MolmoPointer().point(Image.fromarray(camera.rgb),description)
    height,width=camera.depth_m.shape;result=[]
    for u,v in answer.get('points',[]):
        seed=Detection(output_label,1.,(0,0,width,height),None,(round(u),round(v)))
        region=pointed_region(camera,seed,radius=.30)
        if region is not None:result.append(region)
    return result
