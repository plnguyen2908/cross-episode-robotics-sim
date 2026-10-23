"""Measure matched physical traces and export normal-speed A/B videos."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import mujoco


def motion_metrics(rows, addresses):
    if len(rows)<5:return {}
    times=np.array([r['time'] for r in rows])
    q=np.array([r['qpos'] for r in rows])[:,addresses]
    v=np.diff(q,axis=0)/np.diff(times)[:,None]
    acc=np.diff(v,axis=0)/np.diff(times)[1:,None]
    speed=np.linalg.norm(v,axis=1)
    return dict(duration_s=float(times[-1]-times[0]),
                path_length=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum()),
                peak_speed=float(speed.max()),
                rms_acceleration=float(np.sqrt(np.mean(np.sum(acc*acc,axis=1)))),
                p95_acceleration=float(np.percentile(np.linalg.norm(acc,axis=1),95)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path);p.add_argument('smooth',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    results=[]
    for folder in (a.baseline,a.smooth):
        result=json.loads((folder/'comparison.json').read_text())
        rows=json.loads((folder/'trace.json').read_text())
        model=mujoco.MjModel.from_binary_path(str(Path(result['source'])/'model.mjb'))
        base=[model.joint('robot_0/'+n).qposadr[0] for n in ('base_x','base_y')]
        arm=[model.joint('robot_0/fr3_joint'+str(i)).qposadr[0] for i in range(1,8)]
        result['motion_metrics']={}
        # Only contiguous stage groups: never differentiate across omitted time.
        for stage in sorted(set(r['stage'] for r in rows)):
            if 'lift bread' not in stage and not any(x in stage for x in ('drive and turn bread','slide base bread')):continue
            selected=[r for r in rows if r['stage']==stage]
            groups=[]
            for row in selected:
                if not groups or row['time']-groups[-1][-1]['time']>.06:groups.append([])
                groups[-1].append(row)
            result['motion_metrics'][stage]=[motion_metrics(g,arm if 'lift' in stage else base) for g in groups if len(g)>5]
        results.append(result)
    (a.output/'metrics.json').write_text(json.dumps(results,indent=2))
    initial=[]
    for folder in (a.baseline,a.smooth):
        with np.load(folder/'initial_state.npz') as data:initial.append(data['qpos'])
    print('Identical initial qpos:',np.array_equal(*initial),flush=True)
    for title,indices in [('full_sequence',[0,1,2,3]),('navigation',[2]),('pick_and_place',[1,3])]:
        clips=[]
        for index,folder in enumerate((a.baseline,a.smooth)):
            calls=results[index]['calls'];parts=[]
            for part,call_index in enumerate(indices):
                call=calls[call_index]
                clip=a.output/f'{title}_{index}_{part}.mp4';parts.append(clip)
                subprocess.run(['ffmpeg','-v','error','-ss',str(call['start_time']),'-i',str(folder/('review.mp4' if (folder/'review.mp4').exists() else 'composite.mp4')),'-t',str(call['end_time']-call['start_time']),'-an','-c:v','libx264','-preset','veryfast','-crf','24','-y',str(clip)],check=True)
            listing=a.output/f'{title}_{index}_concat.txt'
            listing.write_text(''.join("file '"+str(x.resolve())+"'\n" for x in parts))
            clip=a.output/f'{title}_{index}.mp4';clips.append(clip)
            subprocess.run(['ffmpeg','-v','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-y',str(clip)],check=True)
        output=a.output/f'{title}_comparison.mp4'
        # Retain all three original views; labels distinguish top and bottom.
        from PIL import Image, ImageDraw, ImageFont
        label=a.output/'labels.png'
        im=Image.new('RGBA',(1920,960),(0,0,0,0));draw=ImageDraw.Draw(im)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
        for y,text in [(45,'CURRENT'),(525,results[1]['variant'].upper())]:
            draw.rectangle((0,y,340,y+32),fill='black');draw.text((8,y),text,font=font,fill='white')
        im.save(label)
        filt="[0:v]tpad=stop_mode=clone:stop_duration=300[a];[1:v]tpad=stop_mode=clone:stop_duration=300[b];[a][b]vstack[stack];[stack][2:v]overlay=0:0[v]"
        duration=max(sum(r['calls'][i]['end_time']-r['calls'][i]['start_time'] for i in indices) for r in results)
        subprocess.run(['ffmpeg','-v','error','-i',str(clips[0]),'-i',str(clips[1]),'-i',str(label),'-filter_complex',filt,'-map','[v]','-t',str(duration),'-c:v','libx264','-preset','veryfast','-crf','26','-movflags','+faststart','-y',str(output)],check=True)
        print(output,flush=True)


if __name__=='__main__':main()
