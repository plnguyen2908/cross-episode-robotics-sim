"""Table docking with the carried arm posture in the route layer.

The base-only slice lets the low base pass beneath a table top, and the dock
filter sees only the support patch inside the placement view. A carried
shoulder at table height can then be routed into a part of the top outside
that view; physics stops the base there. Check the retained arm, gripper and
mast against the whole frozen 3D map at the dock and along the route. The
base keeps the base-height slice, so it may still stand beneath an overhang.

The base is also checked against base-height depth seen during the stage,
which the estimate's drift against the frozen map cannot move. No payload
volume enters this layer; physics retains collisions. Frozen map and onboard
depth only, never simulator geometry.
"""
from .basket_docking import FrozenBasketDocking,FrozenBaseFootprintChecker
from .live_base_obstacles import LiveBaseObstacles
from .shaped_docking import ObservedShapeChecker,sweep


class RetainedArmChecker(FrozenBaseFootprintChecker):
    def __init__(self,grid,base_boxes,upper_boxes,initial_pose,margin=.035,live=None):
        super().__init__(grid,base_boxes,initial_pose,margin)
        self.upper=ObservedShapeChecker(grid,upper_boxes,initial_pose,margin=margin,require_observed_support=False)
        self.live=live

    def clear(self,pose):
        if not super().clear(pose):return False
        if not self.upper.clear(pose):
            self.detail=dict(self.upper.detail,reason='frozen_map_arm_collision')
            return False
        return self.live_clear(pose)

    def live_clear(self,pose,points=None):
        """Base footprint against this stage's depth; points limits the test to those."""
        hit=self.live.inside(self.boxes,pose,self.margin,points) if self.live is not None else ()
        self.detail=dict(reason='live_depth_base_collision',part='base_footprint',
            points_map=hit[:8].tolist(),height_band_m=self.height_band) if len(hit) else None
        return not len(hit)

    def live_route_clear(self,pose,route,points=None):
        """A route still to be driven from pose, against this stage's depth only.

        The frozen layers were checked when the route was planned and do not change.
        """
        return all(self.live_clear(p,points) for a,b in zip([pose]+list(route),route) for p in sweep(a,b))


class RetainedArmDocking(FrozenBasketDocking):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.live=LiveBaseObstacles()

    def make_checker(self,pose,q):
        base=super().make_checker(pose,q)
        upper=[b for b in self.geometry.boxes(q) if not b.name.startswith('robot_0/base/')]
        if not upper:raise ValueError('Missing calibrated arm volumes')
        return RetainedArmChecker(self.grid,base.boxes,upper,pose,base.margin,live=self.live)
