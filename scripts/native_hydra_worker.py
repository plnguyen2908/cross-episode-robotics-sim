"""Native Hydra worker: requests contain sensor-recording paths, never env IDs."""
import argparse,json,sys,traceback
from pathlib import Path
import numpy as np
import spark_dsg
from cross_episode_sim.mapping.hydra_mapping import NativeHydraMapper,LABELS
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
mapper=None;count=0
for line in sys.stdin:
    try:
        request=json.loads(line)
        if request.get('close'):
            if mapper:mapper.save(a.output)
            print('HYDRA_RESULT '+json.dumps(dict(ok=True,frames=count)),flush=True)
            break
        with np.load(request['frame']) as f:frame={k:f[k] for k in f.files}
        if mapper is None:mapper=NativeHydraMapper(frame)
        graph=mapper.update(frame);count+=1;names={v:k for k,v in LABELS.items()}
        objects=[dict(id=str(n.id),label=names.get(int(n.attributes.semantic_label),'unknown'),position=n.attributes.position.tolist()) for n in graph.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes]
        rooms=[]
        place_ids={str(n.id) for n in graph.get_layer(spark_dsg.DsgLayers.PLACES).nodes}
        for room in graph.get_layer(spark_dsg.DsgLayers.ROOMS).nodes:
            places=[]
            for child in room.children():
                node=graph.get_node(child)
                if str(node.id) in place_ids:places.append(node.attributes.position.tolist())
            rooms.append(dict(id=str(room.id),position=room.attributes.position.tolist(),places=places))
        print('HYDRA_RESULT '+json.dumps(dict(ok=True,frames=count,objects=objects,rooms=rooms,mesh_vertices=graph.mesh.num_vertices() if graph.has_mesh() else 0)),flush=True)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print('HYDRA_RESULT '+json.dumps(dict(ok=False,error=str(exc))),flush=True)
        break
