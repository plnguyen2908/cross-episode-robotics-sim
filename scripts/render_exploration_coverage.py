"""Post-run room coverage visualization. Never imported by the exploration policy."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
report_path=a.run/'coverage_rescore.json'
if not report_path.exists():report_path=a.run/'report.json'
report=json.loads(report_path.read_text());layout=json.loads((a.source/'base_layout.json').read_text());path=np.load(a.run/'diagnostic_positions.npy')
fig,ax=plt.subplots(figsize=(12,10));scores={s['name']:s for s in report['room_scores']}
for room in layout['rooms']:
    poly=np.asarray(room['floor_polygon']);s=scores[room['name']];visited=s['interior_visit_seconds']>=1.
    ax.add_patch(Polygon(poly,facecolor='#e0eedf' if visited else '#f3dddd',edgecolor='#444444',lw=2))
    center=np.asarray(room['center']);ax.text(*center,f'{room["name"]}\n{100*s["floor_coverage"]:.0f}% floor observed',ha='center',va='center',fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
ax.plot(path[:,0],path[:,1],color='#255eab',lw=1.3,label='Executed robot path (diagnostic ground truth)')
ax.plot(*path[0,:2],'*',ms=15,color='#166d26',label='Start');ax.plot(*path[-1,:2],'X',ms=9,color='#bb2525',label='End')
ax.set_aspect('equal');ax.autoscale_view();ax.margins(.08);ax.legend(loc='upper left',fontsize=8)
ax.set(xlabel='World x (m), diagnostic only',ylabel='World y (m), diagnostic only',title=f'Whole-house exploration: {report["rooms_visited"]}/{report["room_count"]} rooms entered\n{report["status"]} · {report["simulated_seconds"]:.1f} simulated seconds · no room graph supplied to policy')
fig.tight_layout();fig.savefig(a.run/'coverage.png',dpi=160);plt.close(fig)
