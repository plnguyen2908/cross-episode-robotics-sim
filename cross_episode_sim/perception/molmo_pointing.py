"""RGB-only Molmo2 pointing proposals; points never authorize robot motion."""
import re
import time


def parse_points(text, width, height):
    """Molmo2 image points use integer coordinates normalized to 0..1000."""
    result=[]
    for group in re.findall(r'<points?\b[^>]*coords="([0-9\s.,;]+)"',text):
        for _,x,y in re.findall(r'(\d+)\s+(\d{3,4})\s+(\d{3,4})',group):
            x,y=int(x),int(y)
            if 0<=x<1000 and 0<=y<1000:result.append((x*width/1000,y*height/1000))
    return result


class MolmoPointer:
    def __init__(self, model='allenai/Molmo2-4B', device='cuda:0'):
        from .rpc import configured_client
        self.remote=configured_client('molmo')
        if self.remote is not None:return
        import torch
        from transformers import AutoProcessor,AutoModelForImageTextToText
        self.torch=torch
        self.processor=AutoProcessor.from_pretrained(model,trust_remote_code=True)
        self.model=AutoModelForImageTextToText.from_pretrained(model,trust_remote_code=True,torch_dtype=torch.bfloat16).to(device).eval()

    def point(self,image,target):
        from .target_language import visual_target_text
        target=visual_target_text(target)
        if self.remote is not None:
            import numpy as np
            return self.remote.call(image=np.asarray(image),target=target)
        messages=[dict(role='user',content=[dict(type='image',image=image),dict(type='text',text=f'Point to the {target}.')])]
        inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True)
        inputs={k:v.to(self.model.device) for k,v in inputs.items()}
        self.torch.cuda.synchronize();start=time.monotonic()
        with self.torch.inference_mode():out=self.model.generate(**inputs,max_new_tokens=128,do_sample=False)
        self.torch.cuda.synchronize()
        raw=self.processor.tokenizer.decode(out[0,inputs['input_ids'].size(1):],skip_special_tokens=True)
        return dict(raw=raw,points=parse_points(raw,*image.size),wall_seconds=time.monotonic()-start)


class PointingSupervisor:
    """Fresh RGB pointing + instance masks + measured depth, with two-frame consent."""
    def __init__(self,pointer,segmenter,label,period=.8):
        from .visual_targets import VisualTargetMap
        self.pointer=pointer;self.segmenter=segmenter;self.label=label;self.period=period
        self.targets=VisualTargetMap();self.last=-float('inf');self.events=[]

    def refine_mask(self, camera, mask):
        return mask

    def observe(self,packet,tracking):
        import numpy as np
        from PIL import Image
        from .visual_targets import Detection
        if not tracking.accepted or packet.timestamp-self.last<self.period:return
        self.last=packet.timestamp
        cameras=[packet.cameras['navigation']]
        if any(np.linalg.norm(t.position[:2]-tracking.T_map_base[:2,3])<1.1 for t in self.targets.targets):
            cameras.append(packet.cameras['wrist'])
        for camera in cameras:
            result=self.pointer.point(Image.fromarray(camera.rgb),self.label+'s')
            detections=[]
            if result['points']:
                _,masks=self.segmenter.segment(camera.rgb[:,:,::-1].copy())
                masks=[np.asarray(m,bool) for m in masks]
                for x,y in result['points']:
                    u,v=int(x),int(y)
                    choices=[m for m in masks if m.shape==camera.depth_m.shape and m[v,u] and 30<=m.sum()<.15*m.size]
                    if not choices:continue
                    mask=min(choices,key=lambda m:m.sum())
                    mask=self.refine_mask(camera,mask)
                    if mask is None:continue
                    if any(np.array_equal(mask,d.mask) for d in detections):continue
                    yy,xx=np.where(mask)
                    detections.append(Detection(self.label,1.,(float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)),mask=mask))
            self.targets.update(camera,detections,tracking,packet.timestamp)
            self.events.append(dict(timestamp=packet.timestamp,camera=camera.frame_id,raw=result['raw'],wall_seconds=result['wall_seconds'],masks=len(detections),targets=[dict(id=t.track_id,position=t.position.tolist(),last_seen=t.last_seen,observations=t.observations) for t in self.targets.targets]))
