"""Carry departure follows the selected navigation contract, including recovery."""
import numpy as np
from .shaped_docking import ShapedDocking, ObservedShapeChecker, RobotBox
from .observed_undock import ObservedUndock
from .house_exploration import HouseExplorationConfig


class FrozenBaseDepartureDocking(ShapedDocking):
    """Plan the base footprint against every occupied frozen-map column.

    A short reverse/forward departure connects the manipulation dock to the
    circular room-transit region. No fresh depth or held-object 3D veto is added
    to frozen-map navigation. This does not certify manipulator clearance;
    retained posture and payload must still be qualified in physics.
    """
    edge_first=False  # not a pickup among neighbouring instances
    def make_checker(self, pose, q):
        if not hasattr(self.grid, 'fixed_obstacle_points'):
            raise ValueError('Frozen-map departure requires a frozen map')
        self.record_support(pose)
        boxes=[b for b in self.geometry.boxes(q) if b.name.startswith('robot_0/base/')]
        if not boxes:raise ValueError('Missing calibrated base footprint')
        # Extrude the known base footprint over the entire stored obstacle
        # height range: equivalent to a 2D occupied-column sweep, not clearing
        # overhead obstacles or modifying the map to make a route.
        projected=[]
        for box in boxes:
            if not np.allclose(box.axes[:2,2],0.,atol=1e-8):
                raise ValueError('Base footprint must be planar')
            center=box.center.copy();center[2]=self.grid.config.robot_height/2
            half=box.half.copy();half[2]=self.grid.config.robot_height/2
            projected.append(RobotBox('frozen_base_footprint',center,box.axes,half))
        return ObservedShapeChecker(self.grid,projected,pose,
            support_poses=list(getattr(self.grid,'inherited_support_poses',[]))+self.support_poses+[pose])


# A turn in place with a held object is looked ahead by these angles (deg); it is
# refused when any of them puts more mapped points inside the padded robot and
# held-object volumes than the standing pose has.
LOADED_TURN_LOOKAHEAD_DEGREES=(5.,10.,15.,20.,30.)
LOADED_TURN_BACK_SPEED=.10      # m/s, straight back while the turn stays refused
LOADED_TURN_BACK_LIMIT=.60      # m, most extra back-off of one departure
LOADED_TURN_BACK_STEP=.05       # m, the back-off is looked ahead this far
LOADED_TURN_MIN_YAW=1e-4        # rad/s, any turning with the base otherwise at rest is a turn in place
LOADED_TURN_MAX_FORWARD=.02     # m/s


def loaded_turn_blocked(checker,pose,direction):
    """Whether turning in place (direction +1 left, -1 right) brings mapped points into the loaded robot."""
    pose=np.asarray(pose,float);standing=checker.contact_count(pose)
    return any(checker.contact_count(np.r_[pose[:2],pose[2]+direction*np.radians(angle)])>standing
               for angle in LOADED_TURN_LOOKAHEAD_DEGREES)


def straight_back_clear(checker,pose,step=LOADED_TURN_BACK_STEP):
    """Whether moving straight back by `step` brings no further mapped points into the loaded robot."""
    pose=np.asarray(pose,float);back=pose.copy();back[:2]-=step*np.array([np.cos(pose[2]),np.sin(pose[2])])
    return checker.contact_count(back)<=checker.contact_count(pose)


class LoadedTurnGuard:
    """Replace a refused loaded turn in place by a straight back-off.

    Leaving the opened microwave with the food, the retraced approach backed
    0.57 m and turned left in place; the held food met the tip of the open door
    and fell (fix144 call 19; the drink at the open fridge door likewise, call
    27). The approach had been driven with the arm tucked. The robot's own map
    and held-surface estimate showed both contacts: 19 points inside the padded
    volumes standing, 98 after 20 deg of that turn, none after 0.2 m more
    back-off. The decision is made once per sensor packet.
    """
    def __init__(self):
        self.backed=0.;self.speed=0.;self.stamp=None;self.back_off=False;self.announce=False;self.announced=False

    def command(self,forward,yaw,pose,stamp,make_checker,dt=.02,acceleration=.5):
        """Forward speed (m/s, negative) to use instead of the turn, or None to keep the given command.

        The caller zeroes the follower's speed while the back-off lasts, so the
        follower's next command is the first small step of the same turn: any
        turn in place counts, however slow, and the back-off keeps its own
        speed from packet to packet. (With a speed threshold the first replay
        alternated three ticks of slow turning with one tick of back-off and
        still crept into the door.)
        """
        turning=abs(yaw)>LOADED_TURN_MIN_YAW and abs(forward)<LOADED_TURN_MAX_FORWARD
        if stamp!=self.stamp:
            self.stamp=stamp;self.back_off=False
            if turning and self.backed<LOADED_TURN_BACK_LIMIT:
                checker=make_checker()
                self.back_off=bool(loaded_turn_blocked(checker,pose,1. if yaw>0 else -1.) and straight_back_clear(checker,pose))
        if not (self.back_off and self.backed<LOADED_TURN_BACK_LIMIT and abs(forward)<LOADED_TURN_MAX_FORWARD):
            self.speed=0.;return None
        if not self.announced:self.announced=self.announce=True
        self.speed=min(LOADED_TURN_BACK_SPEED,self.speed+acceleration*dt);self.backed+=self.speed*dt
        return -self.speed


class RecordedThenCheckedDeparture:
    """Retrace the approach; when it cannot be retraced, leave by a shape-checked base route.

    The carry away from the microwave stopped at once with
    'recorded_departure_unavailable' (fix137, call 19: the food had just been
    picked after a second look from a slightly moved stand spot) and the food
    was never served. The frozen-map base-footprint departure that fixture
    returns already use is tried before the carry is given up.
    """
    def __init__(self,recorded,checked_factory,payload=None,grid=None):
        self.active=recorded;self.checked_factory=checked_factory;self.switched=False
        self.payload=payload;self.grid=grid;self.turn_guard=LoadedTurnGuard()
    @property
    def follower(self):return self.active.follower
    @property
    def error(self):return self.active.error
    @property
    def docking(self):return self.active.docking
    def step(self,packet,tracking,free,now):
        result=self.active.step(packet,tracking,free,now)
        if not self.switched and self.active.error=='recorded_departure_unavailable' and self.active.follower is None:
            events=list(self.active.docking.events)
            self.active=self.checked_factory();self.switched=True
            self.active.docking.events.extend(events)
            self.active.docking.events.append(dict(event='recorded_departure_unavailable',fallback='shape_checked_base_departure'))
            print('CARRY_DEPARTURE_CHECKED_FALLBACK',flush=True)
            result=self.active.step(packet,tracking,free,now)
        return self._guard_loaded_turn(result,packet,tracking)

    def _guard_loaded_turn(self,result,packet,tracking):
        """Back straight off instead of turning the held object into something the map shows."""
        if self.payload is None or self.grid is None or not tracking.accepted:return result
        if result.get('status') in ('undock_complete',) or self.active.error:return result
        T=tracking.T_map_base;pose=np.r_[T[:2,3],np.arctan2(T[1,0],T[0,0])]
        def checker():
            return self.payload.checker(self.grid,packet.arm.position,pose,
                list(getattr(self.grid,'inherited_support_poses',[]))+[pose])
        command=self.turn_guard.command(result.get('forward',0.),result.get('yaw',0.),pose,packet.timestamp,checker)
        if command is None:return result
        if hasattr(self.active,'velocity'):self.active.velocity[:]=0.
        if self.turn_guard.announce:
            self.turn_guard.announce=False
            self.active.docking.events.append(dict(event='loaded_turn_blocked_back_off',position=pose[:2].tolist(),
                turn='left' if result.get('yaw',0.)>0 else 'right'))
            print('CARRY_LOADED_TURN_BACK_OFF',flush=True)
        return dict(result,forward=float(command),yaw=0.)


def make_carry_departure(grid, payload, *, known_map_only, allow_initial_overlap_exit=False):
    if known_map_only:
        from .recorded_departure import RecordedDeparture
        # No second shape/height-projection veto on a traveled approach.
        recorded=RecordedDeparture(grid)
        geometry=getattr(payload,'geometry',None)
        if geometry is None or not hasattr(grid,'fixed_obstacle_points'):return recorded
        return RecordedThenCheckedDeparture(recorded,lambda:ObservedUndock(FrozenBaseDepartureDocking(grid,geometry)),
            payload=payload if hasattr(payload,'checker') else None,grid=grid)
    from cross_episode_sim.control.observed_payload import PayloadDocking
    return ObservedUndock(PayloadDocking(grid,payload))
