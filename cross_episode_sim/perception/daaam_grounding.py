"""Upstream DAAAM segmentation/tracking with pluggable masked-image VLM labels.

This imports DAAAM services; it is not its complete Hydra scene-graph pipeline.
The lightweight best-view scheduler and VLM transport belong to our integration.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from .visual_targets import Detection, GroundingDinoDetector, box_iou

PROMPT = '''Identify ONLY the red-highlighted object in the first image. The second image shows the same object's isolated crop on gray. Use the scene context to distinguish a shallow, wide bowl from a tall drinking vessel. A bowl has a wide open mouth and no stem; a wine glass has a narrow stem. Do not invent a handle.
First describe its shape and what object it is. Do not guess a hidden object.
Then assign category: cup (including a drinking glass), bowl, dining table,
other (any other identifiable object), or unknown (unclear or mixed region).
Return only JSON with description and category. A region showing several objects
or a fragment of furniture is not an individual vessel.'''   


TABLE_PROMPT = """Identify ONLY the red-highlighted furniture surface using the full scene and its crop.
Distinguish a dining table from a kitchen counter, cabinet, door, floor, or chair.
Small objects or a robot may occlude the furniture; do not label the occluder.
Return only JSON: {"category":"dining table|other|unknown","description":"brief evidence"}.
Choose one category value. Keep the description under 20 words.
Use unknown if the highlighted region does not provide enough evidence."""


def parse_label(text):
    text=text.strip()
    if text.startswith('```'):
        text=text.split('\n',1)[-1].rsplit('```',1)[0].strip()
    value=json.loads(text)
    if not isinstance(value,dict) or value.get('category') not in {'cup','bowl','dining table','other','unknown'}:
        raise ValueError('Invalid VLM category')
    if not isinstance(value.get('description'),str):raise ValueError('Missing description')
    return value


def masked_views(rgb, mask):
    """Highlight the true segmentation mask, retaining context for disambiguation."""
    mask=np.asarray(mask,bool)
    if mask.shape!=rgb.shape[:2] or not mask.any():raise ValueError('Invalid mask')
    overlay=rgb.copy();overlay[mask]=(overlay[mask]*.65+np.array([255,0,0])*.35).astype(np.uint8)
    y,x=np.where(mask);pad=max(16,int(.2*max(np.ptp(x),np.ptp(y))))
    box=(max(0,x.min()-pad),max(0,y.min()-pad),min(rgb.shape[1],x.max()+pad+1),min(rgb.shape[0],y.max()+pad+1))
    full=Image.fromarray(overlay)
    isolated=np.full_like(rgb,127);isolated[mask]=rgb[mask]
    return [full,Image.fromarray(isolated).crop(box)]


class LocalVLM:
    def __init__(self,model='Qwen/Qwen3-VL-8B-Instruct',device='cuda'):
        import torch
        from transformers import AutoProcessor,AutoModelForImageTextToText
        self.torch=torch;self.device=device;self.model_name=model
        self.processor=AutoProcessor.from_pretrained(model)
        self.model=AutoModelForImageTextToText.from_pretrained(model,torch_dtype=torch.bfloat16).to(device).eval()

    def label(self, images):
        return self.label_with_prompt(images,PROMPT)

    def label_with_prompt(self, images, prompt, max_new_tokens=100):
        images=list(images)  # Marked context plus isolated crop; both are actual RGB.
        messages=[{'role':'user','content':[{'type':'image'} for _ in images]+[{'type':'text','text':prompt}]}]
        text=self.processor.apply_chat_template(messages,add_generation_prompt=True)
        inputs=self.processor(text=text,images=images,return_tensors='pt').to(self.device)
        with self.torch.inference_mode():output=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
        raw=self.processor.batch_decode(output[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
        return raw


class EndpointVLM:
    """Optional OpenAI-compatible endpoint; keys are read from environment only."""
    def __init__(self,base_url,model,key_env='VLM_API_KEY'):
        import os
        from openai import OpenAI
        self.client=OpenAI(base_url=base_url,api_key=os.environ.get(key_env,'local'),timeout=60.,max_retries=1)
        self.model_name=model

    def label(self,images):
        import base64,io
        content=[{'type':'text','text':PROMPT}]
        for image in images:
            data=io.BytesIO();image.save(data,format='PNG')
            content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data.getvalue()).decode()}})
        return self.client.chat.completions.create(model=self.model_name,messages=[{'role':'user','content':content}],temperature=0,max_tokens=150).choices[0].message.content


@dataclass
class TrackView:
    seen: int=0
    area: int=0
    images: object=None
    label: object=None
    last_query: float=-float('inf')
    proposal_label: str | None=None


class DaaamGrounder:
    def __init__(self,vlm,checkpoint,output=None,device='cuda',min_observations=2,max_queries_per_frame=2,hybrid=True,categories=('cup','bowl'),detector_model='IDEA-Research/grounding-dino-tiny'):
        self.categories=frozenset(categories)
        if not self.categories or not self.categories <= {'cup','bowl','dining table'}:
            raise ValueError('Unsupported perception categories')
        from daaam.config import SegmentationConfig,TrackingConfig
        from daaam.segmentation.services import SegmentationService
        from daaam.tracking.services import TrackingService
        self.segmenter=SegmentationService(SegmentationConfig(model_name=str(Path(checkpoint).resolve()),model_config_path=None,device=device,min_mask_region_area=100))
        self.make_tracker=lambda:TrackingService(TrackingConfig(device=device,with_reid=False))
        self.proposer=GroundingDinoDetector(device=device,model=detector_model,verify_tables=False) if hybrid else None
        self.map_targets=[];self.map_pose=None;self.guided_proposal_count=0
        self.needs_confirmation=False
        self.trackers={};self.views={};self.vlm=vlm;self.min_observations=min_observations
        self.max_queries=max_queries_per_frame;self.query_count=0;self.errors=[]
        self.output=Path(output) if output else None
        if self.output:self.output.mkdir(parents=True,exist_ok=True)

    def set_map_targets(self,targets,tracking):
        self.map_targets=list(targets) if tracking.accepted else []
        self.map_pose=tracking.T_map_base.copy() if tracking.accepted else None

    def detect_camera(self,camera,timestamp):
        # Upstream FastSAM consumes OpenCV-style numpy images; its masks remain
        # pixel-aligned. VLM views retain original RGB.
        boxes,masks=self.segmenter.segment(camera.rgb[:,:,::-1].copy())
        proposal_labels={}
        if self.proposer is not None:
            proposals=[p for p in self.proposer.detect(camera.rgb) if p.label in self.categories]
            if 'dining table' in self.categories and not any(p.label=='dining table' for p in proposals):
                proposals.extend(p for p in self.proposer.detect(camera.rgb,text='a dining table. a chair. a kitchen counter.') if p.label=='dining table')
            if self.map_pose is not None:
                guided=map_guided_proposals(camera,self.map_targets,self.map_pose,boxes,masks)
                proposals.extend(guided);self.guided_proposal_count+=len(guided)
            boxes,masks,proposal_labels=match_proposals(boxes,masks,proposals)
        tracker=self.trackers.setdefault(camera.frame_id,None)
        if tracker is None:tracker=self.trackers[camera.frame_id]=self.make_tracker()
        tracks=tracker.update(boxes,camera.rgb[:,:,::-1].copy())
        output=[];pending=[]
        for row in tracks:
            index=int(row[7]);key=(camera.frame_id,int(row[4]))
            if index<0 or index>=len(masks):continue
            mask=np.asarray(masks[index],bool)
            if mask.shape!=camera.depth_m.shape or mask.sum()<100:continue
            if camera.robot_mask is not None and np.mean(camera.robot_mask[mask])>.25:continue
            view=self.views.setdefault(key,TrackView());view.seen+=1
            hint=proposal_labels.get(index)
            if hint=='dining table':
                from .surfaces import table_surface_mask
                valid=mask & camera.valid_depth
                if camera.robot_mask is not None:valid &= ~camera.robot_mask
                pose=self.map_pose if self.map_pose is not None else np.eye(4)
                mask=table_surface_mask(camera.depth_m,valid,camera.K,pose@camera.T_base_optical)
                if mask.sum()<30:continue
            if hint!=view.proposal_label:
                view=TrackView(seen=1,proposal_label=hint);self.views[key]=view
            area=int(mask.sum())
            if area>view.area:
                view.area=area;view.images=masked_views(camera.rgb,mask)
            if view.seen>=self.min_observations and (view.label is None or view.label['category']=='unknown') and timestamp-view.last_query>=10.:
                pending.append((key,view))
            output.append((row,mask,view))
        for key,view in sorted(pending,key=lambda item:-item[1].area)[:self.max_queries]:
            view.last_query=timestamp;self.query_count+=1
            prompt=TABLE_PROMPT if view.proposal_label=='dining table' else PROMPT
            raw=self.vlm.label_with_prompt(view.images,prompt) if hasattr(self.vlm,'label_with_prompt') else self.vlm.label(view.images)
            try:view.label=parse_label(raw)
            except (ValueError,TypeError):
                view.label={'category':'unknown','description':'Invalid model response'};self.errors.append({'track':key,'response':raw})
            if self.output:
                prefix=self.output/f'query_{self.query_count:05d}'
                for i,image in enumerate(view.images):image.save(str(prefix)+f'_{i}.png')
                Path(str(prefix)+'.json').write_text(json.dumps(dict(track=key,time=timestamp,prompt=prompt,response=raw,parsed=view.label),indent=2))
        # BotSort can withhold a brand-new track until the next frame. Request
        # inspection from current mask/proposal evidence even before activation,
        # otherwise a moving camera can lose small objects before verification.
        self.needs_confirmation=bool(len(boxes)) and (
            not output or any(view.label is None or view.label['category']=='unknown'
                              for _,_,view in output))
        detections=[]
        for row,mask,view in output:
            if view.label and view.label['category'] in self.categories and (self.proposer is None or view.label['category']==view.proposal_label):
                detections.append(Detection(view.label['category'],float(row[5]),tuple(map(float,row[:4])),mask))
        return detections


def match_proposals(boxes,masks,proposals,min_iou=.35):
    """Keep masks tightly associated with RGB object proposals, not background.

    Reindex detections and masks together for the upstream tracker. Labels are
    hypotheses only; the VLM must independently agree before targets are emitted.
    """
    selected={}
    for proposal in sorted(proposals,key=lambda p:-p.score):
        if not len(boxes):break
        overlaps=np.array([box_iou(row[:4],proposal.box) for row in boxes])
        best=int(np.argmax(overlaps))
        if overlaps[best]<min_iou:continue
        if best in selected and selected[best]!=proposal.label:
            selected[best]=None  # conflicting categories remain unconfirmed
        elif best not in selected:selected[best]=proposal.label
    indices=[i for i,label in selected.items() if label is not None]
    return (np.asarray(boxes)[indices].reshape(-1,6),[masks[i] for i in indices],
            {j:selected[i] for j,i in enumerate(indices)})


def map_guided_proposals(camera,targets,T_map_base,boxes,masks):
    """Propose a CURRENT mask at an estimated mapped object, never invent one.

    Native map position guides attention only. Valid current depth, segmentation
    and the independent VLM label are still required downstream.
    """
    T=np.linalg.inv(T_map_base@camera.T_base_optical);result=[]
    h,w=camera.depth_m.shape
    for target in targets:
        if target.observations<1 or target.label not in {'cup','bowl','dining table'}:continue
        q=T[:3,:3]@target.position+T[:3,3]
        if q[2]<=.1:continue
        pixel=camera.K@q;u,v=np.rint(pixel[:2]/pixel[2]).astype(int)
        if not(0<=u<w and 0<=v<h and camera.valid_depth[v,u]):continue
        if camera.robot_mask is not None and camera.robot_mask[v,u]:continue
        if abs(float(camera.depth_m[v,u])-q[2])>.20:continue
        candidates=[i for i,m in enumerate(masks) if m.shape==(h,w) and m[v,u] and np.count_nonzero(m)>=100]
        if not candidates:continue
        i=min(candidates,key=lambda j:np.count_nonzero(masks[j]))
        result.append(Detection(target.label,float(boxes[i,4]),tuple(map(float,boxes[i,:4])),np.asarray(masks[i],bool)))
    return result
