"""Semantic RGB pointing followed by instance segmentation; no detector weights."""
import numpy as np
from PIL import Image
from .visual_targets import Detection


def free_support_target(receptacle):
    """Ask for exposed support rather than an object near its image center."""
    return (f'center of a visible empty, unobstructed spot on the horizontal '
            f'support surface of the {receptacle}, suitable for placing an object. '
            'Choose the bare surface away from its edges, existing objects, '
            'bowls, cups, and the robot. Do not point to an object on the surface')


class PointMaskDetector:
    """The detection interface backed by Molmo2 points and FastSAM masks.

    RGB stays RGB for Molmo; the upstream FastSAM service takes BGR. A point
    without a valid instance mask is not an object detection. Scores are proposal
    indicators, not calibrated confidence; downstream confirmation remains.
    """
    def __init__(self, categories=('cup','bowl','dining table'), *, pointer=None,
                 segmenter=None, device='cuda', checkpoint='external/DAAAM/checkpoints/fastsam/FastSAM-s.pt',
                 free_support=False):
        from .molmo_pointing import MolmoPointer
        self.categories=tuple(dict.fromkeys(categories))
        self.free_support=free_support
        self.pointer=pointer if pointer is not None else MolmoPointer(device=device)
        if segmenter is None:
            from .rpc import configured_client,RemoteSegmenter
            client=configured_client('fastsam')
            if client is not None:segmenter=RemoteSegmenter(client)
            else:
                from pathlib import Path
                from daaam.config import SegmentationConfig
                from daaam.segmentation.services import SegmentationService
                segmenter=SegmentationService(SegmentationConfig(model_name=str(Path(checkpoint).resolve()),model_config_path=None,device=device,min_mask_region_area=100))
        self.segmenter=segmenter

    def detect(self,rgb,*,masks=None):
        rgb=np.asarray(rgb);h,w=rgb.shape[:2];points=[]
        for label in self.categories:
            target=free_support_target(label) if self.free_support else label
            answer=self.pointer.point(Image.fromarray(rgb),target)
            for point in answer.get('points',[]):
                if len(point)!=2 or not np.isfinite(point).all():continue
                x,y=map(float,point)
                if 0<=x<w and 0<=y<h:points.append((label,int(x),int(y)))
        if not points:return []
        if masks is None:_,masks=self.segmenter.segment(rgb[:,:,::-1].copy())
        masks=[np.asarray(m,bool) for m in masks]
        selected={};seeds={}
        for label,u,v in points:
            candidates=[i for i,m in enumerate(masks) if m.shape==(h,w) and m[v,u] and m.sum()>=30]
            if not candidates:continue
            index=min(candidates,key=lambda i:int(masks[i].sum()))
            # Conflicting semantic labels for one mask require another view.
            if index in selected and selected[index]!=label:selected[index]=None
            elif index not in selected:
                selected[index]=label;seeds[index]=(u,v)
        output=[]
        for index,label in selected.items():
            if label is None:continue
            mask=masks[index];yy,xx=np.where(mask)
            output.append(Detection(label,1.,(float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)),mask,seeds[index]))
        return output
