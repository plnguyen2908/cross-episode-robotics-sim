"""Manifest of all permitted fine-tuning videos; no test trajectories."""
import hashlib
import json
import subprocess
from pathlib import Path


def catalog(root):
    root=Path(root).resolve()
    result=[]
    manifest=root/'manifest.json'
    if manifest.is_file():
        if root.name not in ('train','fine_tuning'):
            raise ValueError('History manifest must be under train or fine_tuning')
        for ep in json.loads(manifest.read_text())['episodes']:
            source=(root/ep['folder']).resolve()
            if not source.is_relative_to(root):raise ValueError('History source outside allowed training root')
            if not json.loads((source/'physics_result.json').read_text()).get('physics_success'):
                raise ValueError('Unsuccessful episode in historical manifest')
            video=source/'composite.mp4'
            result.append({'video_id':f'h{len(result):05d}','task':ep['task'],
                'instruction':(source/'instruction.txt').read_text().strip(),
                'path':str(video),'source':str(source),'sha256':file_hash(video),'seconds':probe(video)})
        if not result:raise ValueError('Empty history manifest')
        return result
    for meta in sorted(root.glob('*/fine_tuning/meta/cross_sim.json')):
        info=json.loads(meta.read_text())
        if info.get('split')!='fine_tuning':raise ValueError('History must be fine_tuning')
        for ep in info['episodes']:
            source=Path(ep['source']).resolve()
            # Export metadata identifies the allowed pool. Never traverse test directories.
            if any(x.startswith('test') for x in source.parts):raise ValueError('Test episode in history metadata')
            video=source/'composite.mp4'
            if not video.is_file():raise FileNotFoundError(video)
            result.append({'video_id':f'h{len(result):05d}','task':info['task'],
                'instruction':(source/'instruction.txt').read_text().strip(),
                'path':str(video),'source':str(source),'sha256':file_hash(video),
                'seconds':probe(video)})
    if not result:raise ValueError('No fine-tuning videos found under '+str(root))
    return result


def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def probe(path):
    return float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration',
        '-of','default=noprint_wrappers=1:nokey=1',str(path)],text=True).strip())


def public_catalog(items):
    return [{k:v for k,v in x.items() if k in ('video_id','task','instruction','seconds')} for x in items]


def frames(items,video_id,times,output):
    item=next((v for v in items if v['video_id']==video_id),None)
    if item is None:raise ValueError('Unknown or unavailable history video')
    paths=[]
    for i,t in enumerate(times):
        if not 0<=t<item['seconds']:raise ValueError('Timestamp outside video duration')
        dest=Path(output)/f'{video_id}_{i}_{t:.3f}.png'
        subprocess.run(['ffmpeg','-v','error','-y','-ss',str(t),'-i',item['path'],
                        '-frames:v','1',str(dest)],check=True,capture_output=True)
        if not dest.exists():raise ValueError('Video produced no frame')
        paths.append(dest)
    return paths
