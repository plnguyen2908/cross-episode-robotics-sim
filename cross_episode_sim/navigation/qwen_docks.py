"""Optional Qwen ranking of sensor-derived docks; never authorizes motion.

The existing planner still checks the full retained robot/payload sweep. IDs
refer only to generated poses; the language model cannot supply coordinates.
"""
import io
import json
import time
import numpy as np
from PIL import Image


def observed_approach_targets(surface, preferred=()):
    """Navigation anchors from measured points, without certifying placement.

    A sparse visible patch is enough to approach. Do not require a full payload
    footprint or an insertion/grasp orientation before seeing the target close.
    Returned anchors are existing observations, never invented free space.
    """
    points=np.asarray(surface,float).reshape(-1,3)
    points=points[np.isfinite(points).all(axis=1)]
    if not len(points):raise ValueError('No observed target for local approach')
    preferred=np.asarray(preferred,float).reshape(-1,3)
    anchors=[]
    for hint in [*preferred,np.median(points,axis=0)]:
        if not np.isfinite(hint).all():continue
        point=points[np.argmin(np.linalg.norm(points-hint,axis=1))]
        if all(np.linalg.norm(point-p)>.03 for p in anchors):anchors.append(point)
    return np.asarray(anchors)


def diverse_indices(candidates, limit=12):
    selected=[]
    for i,g in enumerate(candidates):
        if all(np.linalg.norm(g[:2]-candidates[j][:2])>.12 or
               abs((g[2]-candidates[j][2]+np.pi)%(2*np.pi)-np.pi)>.3 for j in selected):
            selected.append(i)
        if len(selected)>=limit:break
    return selected


def parse_order(response, count):
    text=response.strip()
    if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    row=json.loads(text)
    order=row.get('candidate_ids')
    if not isinstance(order,list) or not order or any(type(i) is not int or not 1<=i<=count for i in order):
        raise ValueError('Qwen must select supplied candidate IDs')
    if len(set(order))!=len(order):raise ValueError('Repeated candidate ID')
    return [i-1 for i in order],str(row.get('reason',''))


def candidate_image(grid, pose, candidates, targets):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    fig=Figure(figsize=(7,6));FigureCanvasAgg(fig);ax=fig.subplots()
    layer=np.zeros(grid.floor_views.shape,dtype=np.uint8)
    layer[grid.floor_views>=grid.config.minimum_floor_views]=1
    layer[(grid.evidence>=2).any(axis=2)]=2
    res=grid.config.resolution;o=grid.origin
    ax.imshow(layer,origin='lower',extent=(o[0],o[0]+layer.shape[1]*res,o[1],o[1]+layer.shape[0]*res),
              cmap=ListedColormap(['#30343b','#d8e4eb','#ec8252']),vmin=0,vmax=2,interpolation='nearest')
    ax.scatter(*pose[:2],s=95,c='#58b8ff',edgecolors='white',zorder=5)
    ax.arrow(*pose[:2],.2*np.cos(pose[2]),.2*np.sin(pose[2]),width=.012,color='#58b8ff',zorder=5)
    for i,(g,t) in enumerate(zip(candidates,targets)):
        ax.scatter(*g[:2],s=30,c='#ffe658',zorder=6)
        ax.arrow(*g[:2],.11*np.cos(g[2]),.11*np.sin(g[2]),width=.005,color='#ffe658',zorder=6)
        ax.annotate(str(i+1),g[:2],xytext=(-11,5),textcoords='offset points',color='white',weight='bold',zorder=7)
        ax.scatter(*t[:2],marker='*',c='#e68bff',s=45,zorder=5)
    extent=np.array([pose[:2]]+[g[:2] for g in candidates]+[t[:2] for t in targets]);lo=extent.min(axis=0)-.65;hi=extent.max(axis=0)+.65
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.set_xlabel('SLAM x (m)');ax.set_ylabel('SLAM y (m)')
    ax.set_title('Observed SLAM map and proposed docking poses')
    fig.text(.07,.025,'Blue: robot   Yellow: dock / facing   Purple: target\nOrange: obstacles   Pale: observed floor   Dark: unknown',fontsize=9)
    fig.tight_layout(rect=[0,.08,1,1]);buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=130);buffer.seek(0)
    return Image.open(buffer).convert('RGB')


def rank_docks(grid, pose, candidates, targets, packet, output, task, vlm=None):
    """Return Qwen-preferred poses followed by geometric fallbacks; log both."""
    from ..perception.daaam_grounding import LocalVLM
    from ..runtime import artifacts
    start=time.monotonic();output.mkdir(parents=True,exist_ok=True)
    candidates=[np.asarray(g) for g in candidates];targets=[np.asarray(t) for t in targets]
    selected=diverse_indices(candidates)
    if not selected:return [],dict(reason='no_geometrically_valid_candidates',qwen_called=False)
    shown=[candidates[i] for i in selected];shown_targets=[targets[i] for i in selected]
    map_image=candidate_image(grid,pose,shown,shown_targets);map_image.save(output/'candidate_map.png')
    # Actual evidence used for this decision, including failed routes, is saved.
    np.savez_compressed(output/'observed_map.npz',origin=grid.origin,resolution=grid.config.resolution,
                        occupied=(grid.evidence>=2).any(axis=2),floor=grid.floor_views,pose=pose,
                        candidates=shown,targets=shown_targets)
    images=[Image.fromarray(packet.cameras[k].rgb) for k in ('navigation','wrist')]+[map_image]
    for im in images:im.thumbnail((768,768))
    metadata=[dict(id=i+1,distance_to_target_m=round(float(np.linalg.norm(g[:2]-t[:2])),2),
        travel_distance_m=round(float(np.linalg.norm(g[:2]-pose[:2])),2)) for i,(g,t) in enumerate(zip(shown,shown_targets))]
    from cross_episode_sim.perception.target_language import visual_target_text
    prompt=('Choose base docking poses for '+visual_target_text(task)+'. Images are navigation RGB, wrist RGB, then an observed SLAM map. '
        'Yellow numbered arrows are proposed base centers and facing; blue is current robot; purple stars are sensed targets. '
        'Orange columns may include furniture at any height; dark is unknown, not certified free. '
        'These candidates passed endpoint geometry checks; A* will check the complete route. '
        'Choose a manipulation dock, not an intermediate approach stop. Prefer target distance around 0.5–0.65 m when available, clear base space, facing the opening/table edge, and arm withdrawal room. '
        'A 0.75–1.0 m target distance is outside that preferred range. Do not choose a farther standoff merely to save a small amount of base travel; prefer the requested manipulation range. '
        'Do not choose a long detour for a tiny gain. Return only JSON {"candidate_ids":[best_id,second_id,third_id],"reason":"brief explanation"}. '
        'Use only provided IDs; choose at most three. Candidate measurements: '+json.dumps(metadata))
    response=(vlm or LocalVLM()).label_with_prompt(images,prompt,max_new_tokens=350)
    try:order,reason=parse_order(response,len(shown));valid=True
    except (ValueError,TypeError,KeyError,json.JSONDecodeError) as exc:order=[];reason=str(exc);valid=False
    chosen=[selected[i] for i in order]
    result=dict(qwen_called=True,valid_response=valid,response=response,reason=reason,
                shown_indices=selected,selected_indices=chosen,candidates=[g.tolist() for g in candidates],
                wall_seconds=time.monotonic()-start)
    (output/'selection.json').write_text(json.dumps(result,indent=2))
    print('QWEN_DOCK_SELECTION',json.dumps(result),flush=True)
    indices=chosen+[i for i in range(len(candidates)) if i not in chosen]
    return [candidates[i] for i in indices],result


def plan_ranked_docks(dock,pose,q,candidates):
    """Try Qwen's first pose; use shared multi-goal search if its route fails."""
    if not candidates:return False
    goal=np.asarray(candidates[0]);checker=dock.make_checker(pose,q)
    heading=np.arctan2(goal[1]-pose[1],goal[0]-pose[0])
    if abs((heading-pose[2]+np.pi)%(2*np.pi)-np.pi)>np.pi/2:heading=(heading+2*np.pi)%(2*np.pi)-np.pi
    direct=[np.r_[pose[:2],heading],np.r_[goal[:2],heading],goal]
    if all(checker.swept(a,b) for a,b in zip([pose]+direct,direct)):
        dock.checker=checker;dock.path=direct;dock.goal=goal
        dock.events.append(dict(event='qwen_selected_direct',goal=goal.tolist()));return True
    # One bounded search for the selected pose, then one shared search for alternatives.
    if dock.plan_candidates(pose,q,[goal],source='qwen_selected_astar'):return True
    from .placement_docks import plan_to_any_dock
    dock.events.append(dict(event='qwen_selected_route_unavailable_fallback'))
    return plan_to_any_dock(dock,pose,q,candidates[1:])
