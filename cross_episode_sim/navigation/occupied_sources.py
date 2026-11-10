"""Carry original occupied cell volumes across coordinate changes.

Rasterization is conservative, but its quantization must not become new measured
geometry on the next handoff. Only newly occupied cells become new sources.
"""
from cross_episode_sim.runtime import artifacts
import itertools
import numpy as np

CORNERS=np.array(list(itertools.product((-.5,.5),repeat=3)))


def cell_sources(grid,indices):
    indices=np.asarray(indices).reshape(-1,3)
    c=grid.config
    centers=np.c_[grid.origin[0]+(indices[:,1]+.5)*c.resolution,
                  grid.origin[1]+(indices[:,0]+.5)*c.resolution,
                  c.min_obstacle_height+(indices[:,2]+.5)*c.height_resolution]
    return centers[:,None,:]+CORNERS[None]*[c.resolution,c.resolution,c.height_resolution]


def source_cells(grid,sources):
    """Yield source IDs and cells overlapping their enclosing AABB.

    Boundary-only touching does not add a neighboring voxel. Rotated sources
    may enlarge a raster once, but the original vertices remain unchanged.
    """
    if not len(sources):return
    c=grid.config;origin=np.r_[grid.origin,c.min_obstacle_height]
    size=np.array([c.resolution,c.resolution,c.height_resolution])
    lo=np.floor((sources.min(1)-origin)/size+1e-8).astype(int)
    hi=np.floor((sources.max(1)-origin)/size-1e-8).astype(int)
    span=np.max(hi-lo,axis=0)
    for offset in itertools.product(*(range(int(n)+1) for n in span)):
        xyz=lo+offset
        valid=(xyz<=hi).all(1)&(xyz>=0).all(1)&(xyz<np.array([grid.evidence.shape[1],grid.evidence.shape[0],grid.evidence.shape[2]])).all(1)
        ids=np.flatnonzero(valid);ijk=xyz[valid][:,[1,0,2]]
        yield ids,ijk


def raster_sources(grid,sources):
    result=np.zeros(grid.evidence.shape,bool)
    for _,ijk in source_cells(grid,sources):result[tuple(ijk.T)]=True
    return result


def save_sources(grid,path):
    occupied=grid.evidence>=2
    sources=getattr(grid,'occupied_sources',np.empty((0,8,3)))
    # Drop fully cleared sources; partially cleared source volumes stay
    # conservative until new measurements resolve the entire original cell.
    retained=np.zeros(len(sources),bool)
    for ids,ijk in source_cells(grid,sources):
        np.logical_or.at(retained,ids,occupied[tuple(ijk.T)])
    sources=sources[retained]
    inherited=raster_sources(grid,sources)
    added=cell_sources(grid,np.argwhere(occupied&~inherited))
    sources=np.concatenate([sources,added])
    artifacts.save_archive(path,vertices=sources)
