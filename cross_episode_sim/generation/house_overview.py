"""Diagnostic house overview from saved initial physics; never a policy input."""
import json
from pathlib import Path


def render_house_overview(folder):
    import mujoco
    import numpy as np
    from PIL import Image
    folder=Path(folder)
    target=folder/'house_overview.png'
    if target.exists():
        return target
    model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
    data=mujoco.MjData(model)
    with np.load(folder/'initial_state.npz') as state:
        mujoco.mj_setState(model,data,state['state'],int(state['state_type']))
    mujoco.mj_forward(model,data)
    # Fit physical floor geoms, including legacy houses without layout metadata.
    boxes=[]
    for g in range(model.ngeom):
        name=model.body(int(model.geom_bodyid[g])).name or ''
        if not name.startswith('floor_'):
            continue
        half=np.abs(data.geom_xmat[g].reshape(3,3)) @ model.geom_aabb[g,3:]
        center=data.geom_xpos[g]+data.geom_xmat[g].reshape(3,3) @ model.geom_aabb[g,:3]
        if min(half[:2])>.3:
            boxes.append((center-half,center+half))
    if not boxes:
        raise ValueError('No house floor geometry to frame')
    lo=np.min([b[0] for b in boxes],axis=0);hi=np.max([b[1] for b in boxes],axis=0)
    cam=mujoco.MjvCamera();cam.lookat[:]=[*(lo[:2]+hi[:2])/2,.4]
    cam.distance=float(max(hi[:2]-lo[:2])*1.4+4)
    cam.azimuth=90;cam.elevation=-58
    opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.sitegroup[:]=0
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,1600)
    model.vis.global_.offheight=max(model.vis.global_.offheight,1000)
    with mujoco.Renderer(model,height=1000,width=1600) as renderer:
        renderer.update_scene(data,camera=cam,scene_option=opt)
        Image.fromarray(renderer.render().copy()).save(folder/'house_overview.tmp.png')
    (folder/'house_overview.tmp.png').replace(target)
    (folder/'house_overview.json').write_text(json.dumps(dict(
        source='model.mjb + initial_state.npz',physical_rerun=False,
        diagnostic_only=True,azimuth=90,elevation=-58,lookat=cam.lookat.tolist(),
        distance=cam.distance),indent=2))
    return target
