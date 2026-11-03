"""Fuse sensor recordings with compiled upstream Hydra (no substitute backend)."""
import argparse,json,time
from pathlib import Path
import numpy as np
import spark_dsg
from tqdm import tqdm
from cross_episode_sim.mapping.hydra_mapping import NativeHydraMapper
p=argparse.ArgumentParser();p.add_argument('--frames',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();files=sorted(a.frames.glob('frame_*.npz'))
if not files:raise RuntimeError('No sensor frames')
a.output.mkdir(parents=True,exist_ok=False)
mapper=NativeHydraMapper(np.load(files[0]));start=time.monotonic()
for path in tqdm(files,desc='Native Hydra mapping'):mapper.update(np.load(path))
mapper.save(a.output)
graph=mapper.pipeline.graph
counts={str(layer.id):layer.num_nodes() for layer in graph.layers}
(a.output/'report.json').write_text(json.dumps(dict(backend='MIT-SPARK/Hydra project/daaam',frames=len(files),layer_node_counts=counts,mesh_vertices=graph.mesh.num_vertices() if graph.has_mesh() else 0,wall_seconds=time.monotonic()-start,pose_source='RGB-D SLAM estimated optical camera pose',scope='Native mapping replay, not navigation success'),indent=2))
