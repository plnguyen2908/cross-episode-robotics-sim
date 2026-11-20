"""Fit appliance controls in onboard metric points, without scene identities."""
import numpy as np
from cross_episode_sim.control.appliance_controls import ObservedControl
from .handle_surface import observed_handle_surface


def refit_tracked_knob(points, previous):
    """Update the indicator while retaining its stationary observed axis.

    Re-fitting a panel and silhouette after wrist motion can attach the track
    to a partially occluded rim. Reuse the measured axis and depth extent;
    estimate only the moving indicator from new RGB-D samples.
    """
    points=np.asarray(points,float);normal=previous.outward
    delta=points-previous.center;signed=delta@normal
    tangent=delta-signed[:,None]*normal
    old=previous.surface-previous.center
    old_depth=old@normal
    radius=np.max(np.linalg.norm(old-old_depth[:,None]*normal,axis=1))+.003
    keep=(np.linalg.norm(tangent,axis=1)<radius)&(signed>old_depth.min()-.004)&(signed<old_depth.max()+.004)
    surface=points[keep]
    front=surface[(surface-previous.center)@normal>=np.percentile(old_depth,70)-.003]
    if len(front)<12:raise ValueError('Tracked knob indicator is occluded')
    centered=front-front.mean(0);flat=centered-np.outer(centered@normal,normal)
    _,s,v=np.linalg.svd(flat,full_matrices=False)
    if s[0]<1.3*max(s[1],1e-9):raise ValueError('Tracked knob indicator is ambiguous')
    up=np.array([0.,0.,1.]);up-=normal*(up@normal);up/=np.linalg.norm(up)
    axis=v[0]
    if axis@up<0:axis=-axis
    angle=float(np.arctan2(axis@np.cross(normal,up),axis@up))
    return ObservedControl(previous.center,normal,surface,angle)


# Percentile along the panel normal above which knob points form the front face.
KNOB_FACE_PERCENTILE=70
# Knob surface trimming (m): applied when points spread past SPREAD from the pointed knob.
KNOB_SURFACE_RADIUS=.035
KNOB_SURFACE_SPREAD=.04

def fit_control(points, target, kind):
    points=np.asarray(points,float);target=np.asarray(target,float)
    if kind=='knob':
        surface,info=observed_handle_surface(points,target,min_protrusion=.004)
        normal=np.asarray(info['panel_normal'])
        # A stove knob is ~5 cm across, but the protruding surface can be the
        # whole control strip (fix183 front-left: 4533 points, +/-16 cm wide).
        # Keep the disc around the pointed knob when the surface spreads wider.
        offset=surface-target;radial=np.linalg.norm(offset-np.outer(offset@normal,normal),axis=1)
        if (radial>KNOB_SURFACE_SPREAD).any() and (radial<=KNOB_SURFACE_RADIUS).sum()>=30:
            surface=surface[radial<=KNOB_SURFACE_RADIUS]
        center=np.median(surface,axis=0)
        up=np.array([0.,0.,1.]);up-=normal*np.dot(up,normal);up/=np.linalg.norm(up)
        side=np.cross(normal,up)
        # A tilted camera samples the upper half more densely. The midpoint of
        # the visible silhouette estimates the axis without that density bias.
        # Use the knob's front face only: the protruding surface also took in
        # the panel rim behind and above it, and from the mast the knob's top
        # (fix182 front-right: +2.4 cm, fix183 front-left: +1.9 cm above the
        # axis; both pinches closed on air). Front-face midpoints were within
        # 0.4 cm of the axis in both views.
        depth=surface@normal;face=surface[depth>=np.percentile(depth,KNOB_FACE_PERCENTILE)]
        outline=face if len(face)>=20 else surface
        for tangent in (up,side):
            projected=outline@tangent
            center+=tangent*((projected.min()+projected.max())/2-center@tangent)
        # Estimate the raised indicator's unoriented long axis in the panel.
        # Circular/occluded surfaces cannot establish ON/OFF orientation.
        # The circular backing disk has no orientation. Fit the raised front
        # half of the observed depth component, which carries the indicator bar.
        projection=surface@normal
        # The front face (same slice as the outline above) shows the bar: on
        # saved views of both knobs at OFF it fitted -0.5 and -1.1 deg with
        # clear elongation, where the front half gave -14 deg or no reading.
        indicator=face if len(face)>=20 else surface[projection>=np.median(projection)]
        centered=indicator-indicator.mean(0);flat=centered-np.outer(centered@normal,normal)
        _,s,v=np.linalg.svd(flat,full_matrices=False)
        angle=None
        if len(s)>1 and s[0]>1.3*max(s[1],1e-9):
            axis=v[0]
            if np.dot(axis,up)<0:axis=-axis
            angle=float(np.arctan2(axis@np.cross(normal,up),axis@up))
        return ObservedControl(center,normal,surface,angle)
    if kind!='button':raise ValueError('Unknown appliance control')
    local=points[np.linalg.norm(points-target,axis=1)<.12]
    if len(local)<30:raise ValueError('Insufficient panel depth')
    rng=np.random.default_rng(0);best=None
    for _ in range(100):
        a,b,c=local[rng.choice(len(local),3,replace=False)]
        normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
        if length<1e-8:continue
        normal/=length
        if abs(normal[2])>.5:continue
        count=int((np.abs((local-a)@normal)<.004).sum())
        if best is None or count>best[0]:best=(count,normal)
    if best is None or best[0]<20:raise ValueError('No observed button panel')
    normal=best[1]
    if normal@(-target)<0:normal=-normal
    surface=local[np.linalg.norm(local-target,axis=1)<.018]
    if len(surface)<5:raise ValueError('Button depth missing')
    return ObservedControl(target,normal,surface)
