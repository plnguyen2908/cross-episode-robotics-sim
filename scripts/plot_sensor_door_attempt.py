"""Visualize observed geometry and executed TCP path, including failed attempts."""
import argparse,json,ast
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
r=json.loads((a.run/'report.json').read_text());pts=np.load(a.run/'initial_observed.npz')['points']
pts=pts[::max(1,len(pts)//18000)]
events=[e for e in r['events'] if 'measured' in e]
actual=np.array([e['measured'] for e in events]);target=np.array([e['target'] for e in events])
fig=plt.figure(figsize=(12,5));ax=fig.add_subplot(121,projection='3d')
ax.scatter(*pts.T,s=1,c=pts[:,2],cmap='Greys',alpha=.15)
ax.plot(*actual.T,'o-',color='tab:blue',label='Measured TCP');ax.plot(*target.T,'--',color='tab:orange',label='Commanded TCP')
if 'solution: [' in (r.get('error') or ''):
    rejected=ast.literal_eval(r['error'].split('solution: ',1)[1])
    if rejected and 'sphere' in rejected[0]:
        sphere=np.array(rejected[0]['sphere']);point=np.array(rejected[0]['point'])
        ax.scatter(*sphere[:3],s=100,c='red',marker='x',label='Blocked robot sphere')
        ax.scatter(*point,s=60,c='purple',label='Observed panel point')
ax.set(xlabel='Base x (m)',ylabel='Base y (m)',zlabel='Height (m)',title='Onboard depth and executed handle motion');ax.legend()
b=fig.add_subplot(122);b.plot([e['time'] for e in events],1000*np.linalg.norm(actual-target,axis=1),'o-')
b.set(xlabel='Simulated seconds',ylabel='TCP target error (mm)',title='Contact tracking');b.grid(alpha=.25)
fig.suptitle('PASS' if r['success'] else 'INCOMPLETE — not a qualified open skill')
fig.tight_layout();fig.savefig(a.run/'observed_motion.png',dpi=160)
