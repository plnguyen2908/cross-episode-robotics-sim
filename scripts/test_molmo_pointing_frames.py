"""Evaluate pointing on clean onboard RGB crops from a recorded robot video."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from cross_episode_sim.perception.molmo_pointing import MolmoPointer
p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--times',type=float,nargs='+',default=[0,10,25,40,50]);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
reader=imageio.get_reader(a.video);fps=reader.get_meta_data()['fps'];pointer=MolmoPointer();results=[]
for t in a.times:
 frame=reader.get_data(round(t*fps));h,w=frame.shape[:2]
 # Original panel is 2x2 nav RGB/depth, wrist RGB/depth. Remove header only.
 image=Image.fromarray(frame[:h//2,:w//2]).crop((0,44,w//2,h//2))
 result=pointer.point(image,'bowls');result['video_seconds']=t;results.append(result)
 draw=ImageDraw.Draw(image)
 for x,y in result['points']:draw.ellipse((x-5,y-5,x+5,y+5),outline='red',width=3)
 image.save(a.output/f'frame_{t:g}.png');(a.output/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(result),flush=True)
reader.close()
