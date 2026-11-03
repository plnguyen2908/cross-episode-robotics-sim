"""Re-anchor measured floor and obstacles at a saved SLAM continuation pose."""
import json
import numpy as np

def restore_observed_map(grid,directory):
    old=np.load(directory/'navigation_map.npz')
    cfg=json.loads((directory/'summary.json').read_text())['config']
    poses=np.load(directory/'slam/estimated_base_poses.npy')
    poses=poses[np.isfinite(poses).all(axis=(1,2))]
    if not len(poses):raise ValueError('No accepted SLAM continuation pose')
    anchor=poses[-1];inverse=np.linalg.inv(anchor)
    # Preserve distant observed rooms when reanchoring into a new local grid.
    # Growing the allocation does not add floor/free-space evidence.
    known=(old['floor_views']>0)|(old['evidence']>=2).any(axis=2)
    rows,cols=np.where(known)
    if len(rows):
        points=np.c_[old['origin_xy'][0]+(cols+.5)*cfg['resolution'],old['origin_xy'][1]+(rows+.5)*cfg['resolution'],np.zeros(len(rows))]
        moved=points@inverse[:3,:3].T+inverse[:3,3]
        grid.ensure_bounds(moved[:,:2],padding_m=float(cfg['resolution']))

    # Only cells with all corners already observed become known floor.
    rr,cc=np.indices(grid.floor_views.shape)
    xy=grid.cell_to_world(np.c_[cc.ravel(),rr.ravel()]);counts=[]
    half=grid.config.resolution/2
    for offset in ((-half,-half),(-half,half),(half,-half),(half,half)):
        world=(xy+offset)@anchor[:2,:2].T+anchor[:2,3]
        ij=np.floor((world-old['origin_xy'])/float(old['resolution'])).astype(int)
        valid=(ij[:,0]>=0)&(ij[:,1]>=0)&(ij[:,0]<old['floor_views'].shape[1])&(ij[:,1]<old['floor_views'].shape[0])
        values=np.zeros(len(xy),np.uint16);values[valid]=old['floor_views'][ij[valid,1],ij[valid,0]];counts.append(values)
    grid.floor_views[:]=np.min(counts,axis=0).reshape(grid.floor_views.shape)
    observations=directory/'floor_observations.npz'
    if observations.exists():
        measured=np.load(observations)
        points=measured['points']@inverse[:3,:3].T+inverse[:3,3]
        counts=measured['counts']
        grid.ensure_bounds(points[:,:2],padding_m=float(cfg['resolution']))
        cells=grid.world_to_cell(points[:,:2])
        np.maximum.at(grid.floor_views,(cells[:,1],cells[:,0]),counts)
        grid.record_floor_samples(points,counts)
    # Preserve occupied voxel extents, not just their centers after rotation.
    ij=np.argwhere(old['evidence']>=2)
    centers=np.c_[old['origin_xy'][0]+(ij[:,1]+.5)*cfg['resolution'],old['origin_xy'][1]+(ij[:,0]+.5)*cfg['resolution'],cfg['min_obstacle_height']+(ij[:,2]+.5)*cfg['height_resolution']]
    for dx in (-.5,0.,.5):
      for dy in (-.5,0.,.5):
       for dz in (-.5,0.,.5):
        p=(centers+np.array([dx*cfg['resolution'],dy*cfg['resolution'],dz*cfg['height_resolution']]))@inverse[:3,:3].T+inverse[:3,3]
        cr=grid.world_to_cell(p[:,:2]);zz=np.floor((p[:,2]-grid.config.min_obstacle_height)/grid.config.height_resolution).astype(int)
        valid=(cr[:,0]>=0)&(cr[:,1]>=0)&(cr[:,0]<grid.evidence.shape[1])&(cr[:,1]<grid.evidence.shape[0])&(zz>=0)&(zz<grid.evidence.shape[2])
        grid.evidence[cr[valid,1],cr[valid,0],zz[valid]]=2
    supports=[np.r_[(inverse@P)[:2,3],np.arctan2((inverse@P)[1,0],(inverse@P)[0,0])] for P in poses]
    inherited=directory/'inherited_support_poses.npy'
    if inherited.exists():
        for p in np.load(inherited):
            P=np.eye(4);c,s=np.cos(p[2]),np.sin(p[2]);P[:2,:2]=[[c,-s],[s,c]];P[:2,3]=p[:2]
            Q=inverse@P;supports.append(np.r_[Q[:2,3],np.arctan2(Q[1,0],Q[0,0])])
    # Deduplicate only identical sampled poses, never enlarge their footprints.
    supports=np.unique(np.asarray(supports),axis=0)
    grid.inherited_support_poses=supports
    return list(supports)
