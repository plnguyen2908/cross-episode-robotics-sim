"""Point to observed empty basket floor using the enclosing instance mask."""
import numpy as np
from PIL import Image
from cross_episode_sim.perception.visual_targets import Detection
from cross_episode_sim.perception.point_masks import PointMaskDetector
from cross_episode_sim.perception.container_masks import enclosing_container_mask

def container_interior_detections(rgb,semantic_label,output_label):
 detector=PointMaskDetector(categories=(semantic_label,));_,masks=detector.segmenter.segment(rgb[:,:,::-1].copy());masks=[np.asarray(v,bool) for v in masks]
 from .basket_floor_point import pointed_basket_floor
 floor=pointed_basket_floor(rgb,semantic_label,output_label,detector.pointer,masks)
 if floor:return floor
 objects=detector.detect(rgb,masks=masks);height,width=rgb.shape[:2];result=[]
 for obj in objects:
  mask=obj.mask
  mask=enclosing_container_mask(mask,masks)
  if not mask.any():continue
  yy,xx=np.where(mask);x0,y0,x1,y1=xx.min(),yy.min(),xx.max()+1,yy.max()+1
  pad=max(5,round(.12*max(x1-x0,y1-y0)));left=max(0,x0-pad);top=max(0,y0-pad);right=min(width,x1+pad);bottom=min(height,y1+pad)
  crop=Image.fromarray(rgb[top:bottom,left:right]);scale=640/max(crop.size);resized=crop.resize((round(crop.width*scale),round(crop.height*scale)))
  answer=detector.pointer.point(resized,'an empty spot on the visible interior bottom of the '+semantic_label+', away from existing objects. Do not point on the bottle, walls, handle, counter or rim. If the bottom is not visible return no points')
  print('INTERIOR_PARENT',semantic_label,int(obj.mask.sum()),int(mask.sum()),answer,flush=True)
  for u,v in answer.get('points',[]):
   u=round(left+u/resized.width*crop.width);v=round(top+v/resized.height*crop.height)
   if 0<=u<width and 0<=v<height and mask[v,u]:result.append(Detection(output_label,obj.score,(float(x0),float(y0),float(x1),float(y1)),mask,(u,v)))
 return result
