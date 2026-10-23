"""Collision-checked forward curves with tangent heading and bounded timing."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from cross_episode_sim.navigation.blended_route import make_segment, wrap


@dataclass
class ForwardCurve:
    curve: object
    parameters: np.ndarray
    times: np.ndarray
    headings: np.ndarray
    start: np.ndarray
    end: np.ndarray
    length: float
    rotation: float
    kind: str = 'forward_curve'

    @property
    def duration(self):
        return float(self.times[-1])

    def pose(self, u):
        s = np.interp(np.clip(u, 0, 1)*self.duration, self.times, self.parameters)
        xy = self.curve(s)
        tangent = self.curve(s, 1)
        reference = np.interp(s, self.parameters, self.headings)
        yaw = reference + wrap(np.arctan2(tangent[1], tangent[0])-reference)
        return np.array([*xy, yaw])


def forward_route(path, pose_clear, *, nav_speed, turn_speed, reverse_first=False,
                  straight_corners=False):
    """Simplify XY, round corners, then check the *new* full-body swept poses.

    No sideways fallback: if rounding fails, retain checked turn/drive corners.
    Return None if even those primitives cannot clear the scene.
    """
    path = np.asarray(path, dtype=float)
    current = path[0].copy()
    prefix = []

    def checked(segment):
        count=max(2, int(np.ceil(segment.length/.01)),
                  int(np.ceil(segment.rotation/np.radians(1))),
                  int(np.ceil(segment.duration/.02)))
        return all(pose_clear(segment.pose(u)) for u in np.linspace(0,1,count+1))

    def turn(start, yaw):
        yaw=start[2]+wrap(yaw-start[2])
        if abs(yaw-start[2])<1e-6:return []
        segment=make_segment(start,start[:2],start[2],yaw,'turn',nav_speed,turn_speed,
                             minimum=max(.3,1.875*abs(yaw-start[2])/turn_speed))
        return [segment] if checked(segment) else None

    moves=[p for p in path[1:] if np.linalg.norm(p[:2]-current[:2])>1e-6]
    if reverse_first and moves:
        target=moves[0]
        segment=make_segment(current,target[:2],current[2],current[2],'reverse',
                             min(.08,nav_speed),turn_speed)
        if not checked(segment):return None
        prefix.append(segment);current=segment.end.copy()
        # Reverse undock is the first translated pose in the original route.
        index=next(i for i,p in enumerate(path) if np.linalg.norm(p[:2]-target[:2])<1e-6)
        path=path[index:]
    points=[current[:2].copy()]
    for p in path[1:]:
        if np.linalg.norm(p[:2]-points[-1])>1e-5:points.append(p[:2].copy())
    final_yaw=path[-1,2]
    if len(points)==1:
        last=turn(current,final_yaw)
        return None if last is None else prefix+last

    def build(points, scale):
        xy=np.asarray(points)
        distances=np.linalg.norm(np.diff(xy,axis=0),axis=1)
        knots=np.r_[0.,np.cumsum(distances)]
        slopes=np.zeros_like(xy)
        slopes[0]=(xy[1]-xy[0])/distances[0]
        slopes[-1]=(xy[-1]-xy[-2])/distances[-1]
        if len(xy)>2:slopes[1:-1]=(xy[2:]-xy[:-2])/(knots[2:]-knots[:-2])[:,None]*scale
        curve=CubicHermiteSpline(knots,xy,slopes)
        params=np.unique(np.r_[np.linspace(0,knots[-1],max(3,int(knots[-1]/.005)+1)),knots])
        tangent=curve(params,1)
        if np.min(np.linalg.norm(tangent,axis=1))<1e-5:return None
        headings=np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]))
        headings+=current[2]+wrap(headings[0]-current[2])-headings[0]
        ds=np.linalg.norm(np.diff(curve(params),axis=0),axis=1)
        dy=np.abs(np.diff(headings))
        curvature=dy/np.maximum(ds,1e-9)
        speed=np.full(len(params),nav_speed)
        # Slow translation through bends as well as limiting yaw rate: the
        # physical XY servos lag more than yaw, so fast curves otherwise skid.
        limits=np.minimum(min(turn_speed, .4)/np.maximum(curvature,1e-9),
                          np.sqrt(.04/np.maximum(curvature,1e-9)))
        speed[:-1]=np.minimum(speed[:-1],limits)
        speed[1:]=np.minimum(speed[1:],limits)
        speed[0]=speed[-1]=0.
        for i in range(1,len(speed)):speed[i]=min(speed[i],np.sqrt(speed[i-1]**2+2*.5*ds[i-1]))
        for i in range(len(speed)-2,-1,-1):speed[i]=min(speed[i],np.sqrt(speed[i+1]**2+2*.5*ds[i]))
        times=np.r_[0.,np.cumsum(2*ds/np.maximum(speed[:-1]+speed[1:],1e-8))]
        seg=ForwardCurve(curve,params,times,headings,np.r_[xy[0],headings[0]],
                         np.r_[xy[-1],headings[-1]],float(ds.sum()),float(dy.sum()))
        first=turn(current,headings[0])
        last=turn(seg.end,final_yaw)
        if first is None or last is None or not checked(seg):return None
        return prefix+first+[seg]+last

    # Try the strongest shortcut first, then progressively retain more waypoints.
    candidate=build([points[0],points[-1]],1.)
    if candidate is not None:return candidate
    simplified=[points[0]]
    index=0
    while index<len(points)-1:
        chosen=index+1
        for j in range(len(points)-1,index+1,-1):
            a,b=points[index],points[j]
            yaw=np.arctan2(b[1]-a[1],b[0]-a[0])
            yaw=current[2]+wrap(yaw-current[2])
            count=max(2,int(np.ceil(np.linalg.norm(b-a)/.02)))
            if all(pose_clear(np.r_[a+u*(b-a),yaw]) for u in np.linspace(0,1,count+1)):
                chosen=j;break
        simplified.append(points[chosen]);index=chosen
    if straight_corners:
        # Shortcut complete turn+drive sweeps, not only straight XY chords.
        # A chord-only shortcut can fail its endpoint turn and force the entire
        # route back to every grid corner. Keep only shortcuts whose chassis
        # and retained arm clear both the turn and translated sweep.
        cursor = current.copy()
        index = 0
        direct = list(prefix)
        while index < len(points)-1:
            accepted = False
            for j in range(len(points)-1, index, -1):
                yaw = np.arctan2(points[j][1]-cursor[1], points[j][0]-cursor[0])
                turns = turn(cursor, yaw)
                if turns is None:
                    continue
                start = turns[-1].end if turns else cursor
                distance = np.linalg.norm(points[j]-start[:2])
                drive = make_segment(start, points[j], start[2], start[2],
                    'drive', nav_speed, turn_speed,
                    minimum=max(.3, 1.875*distance/nav_speed, np.sqrt(6*distance/.5)))
                if not checked(drive):
                    continue
                if j == len(points)-1 and turn(drive.end, final_yaw) is None:
                    continue
                direct.extend(turns)
                direct.append(drive)
                cursor = drive.end.copy()
                index = j
                accepted = True
                break
            if not accepted:
                break
        if index == len(points)-1:
            last = turn(cursor, final_yaw)
            if last is not None:
                return direct+last
        # The XY servo can lag curved headings even with an empty hand.
        # Long curved routes accumulate lateral slip at low nominal speed.
        # Use the simplified A* route with checked turn/drive corners,
        # avoiding both lateral travel and dozens of grid turns.
        origin=current.copy()
        for waypoints in (simplified, points):
            current=origin.copy();result=list(prefix);valid=True
            for target in waypoints[1:]:
                yaw=np.arctan2(target[1]-current[1],target[0]-current[0])
                turns=turn(current,yaw)
                if turns is None:valid=False;break
                result.extend(turns)
                if turns:current=turns[-1].end.copy()
                distance=np.linalg.norm(target-current[:2])
                segment=make_segment(current,target,current[2],current[2],'drive',nav_speed,turn_speed,
                    minimum=max(.3,1.875*distance/nav_speed,np.sqrt(6*distance/.5)))
                if not checked(segment):valid=False;break
                result.append(segment);current=segment.end.copy()
            last=turn(current,final_yaw) if valid else None
            if last is not None:return result+last
        return None
    for scale in (1.,.5,.25):
        candidate=build(simplified,scale)
        if candidate is not None:return candidate
    for scale in (1.,.5,.25):
        candidate=build(points,scale)
        if candidate is not None:return candidate
    # A curve may not fit at a tight corner; checked forward primitives still can.
    result=list(prefix)
    for target in points[1:]:
        yaw=np.arctan2(target[1]-current[1],target[0]-current[0])
        turns=turn(current,yaw)
        if turns is None:return None
        result.extend(turns)
        if turns:current=turns[-1].end.copy()
        distance=np.linalg.norm(target-current[:2])
        segment=make_segment(current,target,current[2],current[2],'drive',nav_speed,turn_speed,
                             minimum=max(.3,1.875*distance/nav_speed, np.sqrt(6*distance/.5)))
        if not checked(segment):return None
        result.append(segment);current=segment.end.copy()
    last=turn(current,final_yaw)
    return None if last is None else result+last
