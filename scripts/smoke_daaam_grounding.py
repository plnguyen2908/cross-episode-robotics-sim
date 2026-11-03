"""Real simulator camera frames → upstream segmentation/tracking → local VLM."""
import argparse,json
from contextlib import nullcontext
from pathlib import Path
import numpy as np
from cross_episode_sim.perception.daaam_grounding import DaaamGrounder,LocalVLM
from cross_episode_sim.sensors import OnboardSensorInterface
from cross_episode_sim.slam import RGBDMapper
from cross_episode_sim.mapping.hydra_mapping import HydraFrameWriter
from cross_episode_sim.perception.visual_targets import VisualTargetMap
from cross_episode_sim.control.onboard_base import OnboardBaseActuator
from scripts.demo_local_scan import setup

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--frames',type=int,default=10)
p.add_argument('--native-hydra',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
grounder=DaaamGrounder(LocalVLM(),'external/DAAAM/checkpoints/fastsam/FastSAM-s.pt',a.output/'queries')
m,d,_=setup(a.source);base=OnboardBaseActuator(m,d);results=[];slam=RGBDMapper();targets=VisualTargetMap();hydra_frames=HydraFrameWriter(a.output/'hydra_frames')
from cross_episode_sim.mapping.hydra_process import NativeHydraProcess
live_hydra=NativeHydraProcess(a.output/'native_hydra') if a.native_hydra else nullcontext()
hydra_updates=[]
with live_hydra as native,OnboardSensorInterface(m,d) as sensors:
    for i in range(a.frames):
        packet=sensors.capture();tracking=slam.update(packet.cameras['navigation'],packet.timestamp)
        for name,camera in packet.cameras.items():
            detections=grounder.detect_camera(camera,packet.timestamp)
            targets.update(camera,detections,tracking,packet.timestamp)
            if name=='navigation':
                hydra_frames.update(camera,tracking,detections,packet.timestamp)
                if native:hydra_updates.append(native.update(camera,tracking,detections,packet.timestamp))
            results.append(dict(frame=i,camera=name,detections=[dict(label=x.label,box=x.box,mask_pixels=int(x.mask.sum())) for x in detections]))
        for _ in range(5):base.step(0.,0.)
report=dict(native_hydra_updates=hydra_updates,accepted_3d_targets=[dict(label=t.label,position=t.position.tolist(),observations=t.observations) for t in targets.targets],vlm_queries=grounder.query_count,parse_errors=grounder.errors,results=results,
            upstream_services=['daaam.segmentation.services.SegmentationService','daaam.tracking.services.TrackingService'],
            scope='Stationary real-camera perception smoke test, not cross-room navigation or task success.')
(a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
