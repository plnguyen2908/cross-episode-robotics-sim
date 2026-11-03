"""Build navigation evidence from saved RGB-D + causal SLAM estimates only."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from cross_episode_sim.navigation.observed_map import ObservedNavigationMap,MapConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stream',type=Path,required=True)
    p.add_argument('--tracking',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frame-stride',type=int,default=4)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    calibration=json.loads((a.stream/'calibration.json').read_text())
    depth=np.load(a.stream/'depth.npy',mmap_mode='r');masks=np.load(a.stream/'robot_mask.npy',mmap_mode='r')
    poses=np.load(a.tracking/'all_frame_estimates.npy');times=np.load(a.stream/'timestamps.npy')
    statuses=json.loads((a.tracking/'tracking.json').read_text())
    if len(statuses)!=len(poses) or len(poses)>len(depth):raise ValueError('Tracking and sensor stream lengths disagree')
    if not np.allclose([s['timestamp'] for s in statuses],times[:len(statuses)],atol=1e-6,rtol=0):raise ValueError('Tracking timestamps do not match sensor stream')
    mapper=ObservedNavigationMap()
    start=time.monotonic()
    for i in tqdm(range(0,len(poses),a.frame_stride),desc='Observed navigation map'):
        camera=SimpleNamespace(depth_m=depth[i],valid_depth=np.isfinite(depth[i]),robot_mask=masks[i],
            K=np.array(calibration['K']),T_base_optical=np.array(calibration['T_base_optical']))
        tracking=SimpleNamespace(timestamp=times[i],accepted=statuses[i]['accepted'],T_map_base=poses[i])
        mapper.update(camera,tracking)
    report=mapper.save(a.output)
    report.update(build_wall_seconds=time.monotonic()-start,frame_stride=a.frame_stride,sensor_stream=str(a.stream),tracking_source=str(a.tracking))
    (a.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    layers=mapper.layers();known=layers.floor_observed|layers.occupied
    yy,xx=np.where(known)
    if not len(xx):raise RuntimeError('No observed map evidence')
    pad=12;x0=max(0,xx.min()-pad);x1=min(known.shape[1],xx.max()+pad);y0=max(0,yy.min()-pad);y1=min(known.shape[0],yy.max()+pad)
    extent=[mapper.origin[0]+x0*.1,mapper.origin[0]+x1*.1,mapper.origin[1]+y0*.1,mapper.origin[1]+y1*.1]
    region=np.s_[y0:y1,x0:x1]
    semantic=np.zeros(known.shape,np.uint8);semantic[layers.floor_observed]=1;semantic[layers.occupied]=2
    candidates=np.zeros(known.shape,np.uint8);candidates[layers.candidate_free]=1;candidates[layers.costmap==100]=2;candidates[layers.strict_free]=3
    fig,axes=plt.subplots(1,3,figsize=(17,6),sharex=True,sharey=True)
    axes[0].imshow(semantic[region],origin='lower',extent=extent,cmap=ListedColormap(['#a9afb6','#eaf2e4','#b84b45']),vmin=0,vmax=2,interpolation='nearest')
    axes[0].set_title('Observed floor and obstacles\nGray unknown · pale floor · red obstacles')
    image=axes[1].imshow(np.where(layers.floor_observed,layers.clearance_fraction,np.nan)[region],origin='lower',extent=extent,vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
    axes[1].set_facecolor('#a9afb6');axes[1].set_title('Observed vertical-clearance fraction\nMissing height bands remain unknown')
    fig.colorbar(image,ax=axes[1],fraction=.035,pad=.02)
    axes[2].imshow(candidates[region],origin='lower',extent=extent,cmap=ListedColormap(['#a9afb6','#e8c66a','#df8b83','#42996b']),vmin=0,vmax=3,interpolation='nearest')
    axes[2].set_title('Robot-center clearance\nYellow floor-only candidate · green strict\nRed inflated obstacle · gray unknown/insufficient')
    good=np.isfinite(poses).all(axis=(1,2));path=poses[good]
    for ax in axes:
        ax.plot(path[:,0,3],path[:,1,3],color='#243b80',lw=1,label='Estimated recorded path (not a planned route)')
        ax.set(xlabel='Map x (m)',ylabel='Map y (m)');ax.set_aspect('equal')
    axes[0].legend(loc='lower left',fontsize=7)
    fig.suptitle('Step 3: onboard-only navigation evidence — no simulator map or ground-truth poses',fontsize=13)
    fig.tight_layout();fig.savefig(a.output/'navigation_map.png',dpi=150,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
