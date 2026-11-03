"""Re-evaluate RGB perception on saved sensor packets, with no env geometry."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw
from tqdm import tqdm
from cross_episode_sim.perception.daaam_grounding import DaaamGrounder,LocalVLM
from cross_episode_sim.perception.visual_targets import VisualTargetMap
from cross_episode_sim.mapping.hydra_mapping import prepare_frame
p=argparse.ArgumentParser();p.add_argument('--frames',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--indices',type=int,nargs='*')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);(a.output/'frames').mkdir()
grounder=DaaamGrounder(LocalVLM(),'external/DAAAM/checkpoints/fastsam/FastSAM-s.pt',a.output/'queries',categories=('cup','bowl','dining table'));targets=VisualTargetMap();rows=[]
files=sorted(a.frames.glob('frame_*.npz'))
if a.indices is not None:files=[files[i] for i in a.indices]
for path in tqdm(files,desc='Sensor perception regression'):
    with np.load(path) as f:frame={k:f[k] for k in f.files}
    camera=SimpleNamespace(rgb=frame['rgb'],depth_m=frame['depth'],valid_depth=frame['depth']>0,robot_mask=frame['depth']==0,K=frame['K'],T_base_optical=frame['T_map_optical'],frame_id='navigation')
    tracking=SimpleNamespace(accepted=True,T_map_base=np.eye(4));stamp=float(frame['timestamp_ns'])*1e-9
    detections=grounder.detect_camera(camera,stamp);updated=targets.update(camera,detections,tracking,stamp)
    row=dict(frame=path.name,timestamp=stamp,labels=[d.label for d in detections],updated_ids=[t.track_id for t in updated]);rows.append(row)
    np.savez_compressed(a.output/'frames'/path.name,**prepare_frame(camera,tracking,detections,stamp))
    im=Image.fromarray(frame['rgb']);draw=ImageDraw.Draw(im)
    for d in detections:draw.rectangle(d.box,outline='lime',width=2);draw.text(d.box[:2],d.label,fill='lime')
    im.save(a.output/path.with_suffix('.png').name)
    tqdm.write(json.dumps(row))
(a.output/'report.json').write_text(json.dumps(dict(frames=rows,targets=[dict(id=t.track_id,label=t.label,position=t.position.tolist(),observations=t.observations,last_seen=t.last_seen) for t in targets.targets]),indent=2))
