"""Re-anchor within-episode semantic observations using saved SLAM estimates."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from cross_episode_sim.perception.visual_targets import VisualTarget


def load_memory(source):
    source=Path(source);path=source/'semantic_memory.json'
    if not path.exists():return dict(rooms=[],tables=[])
    memory=json.loads(path.read_text())
    poses=np.load(source/'slam/estimated_base_poses.npy')
    poses=poses[np.isfinite(poses).all(axis=(1,2))]
    if not len(poses):raise ValueError('Semantic continuation needs an accepted SLAM pose')
    inverse=np.linalg.inv(poses[-1])
    def transform(points):
        points=np.asarray(points,float).reshape(-1,3)
        return (points@inverse[:3,:3].T+inverse[:3,3]).tolist()
    for room in memory['rooms']:room['position']=transform([room['position']])[0]
    for table in memory['tables']+memory.get('objects',[]):
        table['position']=transform([table['position']])[0]
        table['surface_points']=transform(table.get('surface_points',[]))
        for item in table.get('history',[]):item['position']=transform([item['position']])[0]
    return memory


def save_memory(output,memory,*,room_events=(),targets=()):
    rooms=list(memory.get('rooms',[]))
    for event in room_events:
        if event.get('confirmed_label','unknown')=='unknown':continue
        position=event['estimated_position']
        label=event['confirmed_label']
        if any(r['label']==label and np.linalg.norm(np.asarray(r['position'])-position)<.5 for r in rooms):continue
        rooms.append(dict(label=label,position=position,timestamp=event['timestamp']))
    tables=list(memory.get('tables',[]))
    observed=[]
    for target in targets:
        if target.label!='dining table' or target.observations<2:continue
        value=asdict(target);value['position']=target.position.tolist();observed.append(value)
    if observed:tables=observed
    objects=list(memory.get('objects',[]))
    for target in targets:
        if target.label not in {'cup','bowl'} or target.observations<1:continue
        value=asdict(target);value['position']=target.position.tolist()
        nearby=[i for i,item in enumerate(objects) if item['label']==target.label and np.linalg.norm(np.asarray(item['position'])-target.position)<.25]
        if nearby:
            index=min(nearby,key=lambda i:np.linalg.norm(np.asarray(objects[i]['position'])-target.position))
            if value['last_seen']>=objects[index]['last_seen']:objects[index]=value
        else:objects.append(value)
    Path(output,'semantic_memory.json').write_text(json.dumps(dict(rooms=rooms,tables=tables,objects=objects,provenance='Onboard visual labels/depth and estimated SLAM poses only'),indent=2))


def table_targets(memory):
    return [VisualTarget(**{**value,'track_id':index,'position':np.asarray(value['position']),'navigation_memory':True}) for index,value in enumerate(memory.get('tables',[]))]


def object_hypotheses(memory):
    """Old sensor sightings guide current masks, never count as fresh targets."""
    return [VisualTarget(**{**value,'track_id':index,'position':np.asarray(value['position']),'navigation_memory':True}) for index,value in enumerate(memory.get('objects',[]))]
