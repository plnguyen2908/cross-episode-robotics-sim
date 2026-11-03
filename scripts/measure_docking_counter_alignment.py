"""Offline diagnostic: compare final heading to the observed cabinet-face normal.

Reads recorded depth/SLAM only; never feeds measurements back to execution.
"""
import json,math
from pathlib import Path
import numpy as np
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('suite',type=Path);args=p.parse_args();results=[]
for folder in sorted(args.suite.iterdir()):
 p=folder/'report.json'
 if not p.exists():continue
 r=json.loads(p.read_text())
 if not r.get('docking_success'):continue
 target=np.array(r['selected_object']['position']);files=sorted((folder/'hydra_frames').glob('frame_*.npz'))
 f=np.load(files[-1]);d=f['depth'];K=f['K'];T=f['T_map_optical'];v,u=np.mgrid[0:d.shape[0]:3,0:d.shape[1]:3];z=d[::3,::3].ravel();valid=np.isfinite(z)&(z>0);u=u.ravel()[valid];v=v.ravel()[valid];z=z[valid];xyz=np.c_[(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z];pts=xyz@T[:3,:3].T+T[:3,3]
 pts=pts[(pts[:,2]>.3)&(pts[:,2]<target[2]-.25)&(np.linalg.norm(pts[:,:2]-target[:2],axis=1)<.85),:2]
 if len(pts)<100:
  results.append(dict(case=folder.name,status='insufficient_face_points'));continue
 rng=np.random.default_rng(0);best=None
 for _ in range(300):
  a,b=pts[rng.choice(len(pts),2,replace=False)];t=b-a
  if np.linalg.norm(t)<.25:continue
  normal=np.array([-t[1],t[0]])/np.linalg.norm(t);mask=np.abs((pts-a)@normal)<.015
  if best is None or mask.sum()>best.sum():best=mask
 if best is None or best.sum()<100:
  results.append(dict(case=folder.name,status='face_unverified'));continue
 _,_,vt=np.linalg.svd(pts[best]-pts[best].mean(0),full_matrices=False);normal=vt[-1];base=np.load(folder/'slam/estimated_base_poses.npy')[-1];direction=target[:2]-base[:2,3]
 if normal@direction<0:normal=-normal
 yaw=math.atan2(base[1,0],base[0,0]);angle=math.atan2(normal[1],normal[0]);err=math.degrees(math.atan2(math.sin(yaw-angle),math.cos(yaw-angle)))
 results.append(dict(case=folder.name,status='estimated' if best.mean()>=.5 else 'weak_plane_consensus',normal_error_deg=err,inlier_fraction=float(best.mean()),points=len(pts),normal_yaw_deg=math.degrees(angle),frame=str(files[-1]),diagnostic_frontal_within_5deg=bool(best.mean()>=.5 and abs(err)<=5.)))
(args.suite/'counter_alignment.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
