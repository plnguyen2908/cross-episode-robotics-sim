"""Re-anchor measured floor and obstacles at a saved SLAM continuation pose."""
from cross_episode_sim.runtime import artifacts
import json
import numpy as np

def restore_observed_map(grid,directory):
    old=artifacts.load_array(directory/'navigation_map.npz')
    cfg=json.loads(artifacts.read_text(directory/'summary.json'))['config']
    poses=artifacts.load_array(directory/'slam/estimated_base_poses.npy')
    poses=poses[np.isfinite(poses).all(axis=(1,2))]
    if not len(poses):raise ValueError('No accepted SLAM continuation pose')
    if artifacts.owner(directory) is not None:
        # A frozen map in the persistent frame is copied exactly, not repeatedly
        # rasterized after every camera/arm/navigation stage.
        # Array indices retain their source voxel size. Keeping a caller's
        # different height resolution would move all obstacles vertically.
        from .observed_map import MapConfig
        grid.config=MapConfig(**cfg)
        grid.origin=np.asarray(old['origin_xy']).copy()
        grid.floor_views=np.asarray(old['floor_views']).copy()
        grid.evidence=np.asarray(old['evidence']).copy()
        source_path=directory/'occupied_sources.npz'
        if artifacts.exists(source_path):grid.occupied_sources=artifacts.load_array(source_path)['vertices'].copy()
        observations=directory/'floor_observations.npz'
        if artifacts.exists(observations):
            measured=artifacts.load_array(observations)
            grid.record_floor_samples(measured['points'],measured['counts'])
        supports=[np.r_[P[:2,3],np.arctan2(P[1,0],P[0,0])] for P in poses]
        inherited=directory/'inherited_support_poses.npy'
        if artifacts.exists(inherited):supports.extend(artifacts.load_array(inherited))
        grid.inherited_support_poses=np.unique(np.asarray(supports).reshape(-1,3),axis=0)
        return list(grid.inherited_support_poses)
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
    if artifacts.exists(observations):
        measured=artifacts.load_array(observations)
        points=measured['points']@inverse[:3,:3].T+inverse[:3,3]
        counts=measured['counts']
        grid.ensure_bounds(points[:,:2],padding_m=float(cfg['resolution']))
        cells=grid.world_to_cell(points[:,:2])
        np.maximum.at(grid.floor_views,(cells[:,1],cells[:,0]),counts)
        grid.record_floor_samples(points,counts)
    # Transform original cell volumes, never last stage's inflated raster.
    from .occupied_sources import cell_sources,raster_sources
    source_path=directory/'occupied_sources.npz'
    if artifacts.exists(source_path):
        sources=artifacts.load_array(source_path)['vertices']
    else:
        # Legacy artifacts have no provenance; preserve their extents once.
        from types import SimpleNamespace
        source_grid=SimpleNamespace(origin=old['origin_xy'],config=SimpleNamespace(**cfg))
        sources=cell_sources(source_grid,np.argwhere(old['evidence']>=2))
    sources=sources@inverse[:3,:3].T+inverse[:3,3]
    if len(sources):grid.ensure_bounds(sources[:,:,:2].reshape(-1,2),padding_m=grid.config.resolution)
    grid.occupied_sources=sources
    grid.evidence[raster_sources(grid,sources)]=2
    supports=[np.r_[(inverse@P)[:2,3],np.arctan2((inverse@P)[1,0],(inverse@P)[0,0])] for P in poses]
    inherited=directory/'inherited_support_poses.npy'
    if artifacts.exists(inherited):
        for p in artifacts.load_array(inherited):
            P=np.eye(4);c,s=np.cos(p[2]),np.sin(p[2]);P[:2,:2]=[[c,-s],[s,c]];P[:2,3]=p[:2]
            Q=inverse@P;supports.append(np.r_[Q[:2,3],np.arctan2(Q[1,0],Q[0,0])])
    # Deduplicate only identical sampled poses, never enlarge their footprints.
    supports=np.unique(np.asarray(supports),axis=0)
    grid.inherited_support_poses=supports
    return list(supports)
