"""Where a held object sits in the hand, from the wrist camera: our live view vs the past robot's view.

A remembered place releases the object where the past robot's hand released ITS object. When our object sits
differently in the hand, the base shifts by that difference first (memory_navigation.shift_base) so the object -
not the hand - lands where the past object landed.

Live side: the onboard wrist camera (depth + the robot-geometry mask from encoder FK). Past side: the past robot's
recorded wrist RGB-D frames (memory/wrist_memory) and its recorded commands (forward kinematics of the commanded
arm joints). Never a past episode's simulator scene (object poses, shapes, contacts); no simulator truth of the
current episode either.
"""
import json
from pathlib import Path

import numpy as np

NEAR_M = .32        # held-object pixels: closer than this to the wrist camera (the counter/fixture is farther)
MIN_POINTS = 150    # fewer object pixels on either side: no measurement -> no shift
PLACE_MAX_DZ_M = .06   # in-hand vertical difference at the release beyond this: memory refuses the place
PLACE_DZ_GATED = ('kitchen_cabinet',)  # shelves: a low release hits the shelf edge (store run 6); a microwave / drawer
                                       # floor just takes the object a little earlier (meal run 10: gate refused, baseline place failed)
FLIP_MIN_X_M = .03
FLIP_LOCATIONS = ('kitchen_cabinet', 'cup_drawer', 'microwave', 'burner_1', 'burner_2')  # bowls/cups (round) and the food box (180-deg symmetric): meal run 13, food gripped from its other end landed tilted on the turntable edge and did not heat; pots (round, 2 handles) on the burners: warm_two_pots_000033 pots_full10, the pot gripped from its other rim side was shifted 13.6 cm instead and released on the stove edge
PLACE_SHIFT = False   # in-hand offsets are measured and logged only: a held object of another shape/size sits differently
                      # and the fixture interior tolerates a few cm (run 7 tests the place unshifted)
FINGER_DEPTH_M = .13  # the open fingers' far ends in the wrist view; the object is the nearest surface beyond them
PICK_NAV_VIEW = False  # docked at a fridge shelf the mast camera cannot see the object (shelf above blocks it; run 5)


def nearest_object(cam, rows=(.25, .7), cols=(.3, .7), band=.06):
    """the object in front of the open fingers: the nearest depth surface beyond the fingers in the central wrist view
    (the conservative robot-geometry mask also covers an object between the fingers, so depth separates them here);
    median point in the base frame"""
    z = np.asarray(cam.depth_m, float); h, w = z.shape; win = np.zeros_like(z, bool)
    win[int(rows[0]*h):int(rows[1]*h), int(cols[0]*w):int(cols[1]*w)] = True
    ok = win & np.isfinite(z) & (z > FINGER_DEPTH_M)
    if ok.sum() < MIN_POINTS:return None, int(ok.sum())
    ok &= z < np.percentile(z[ok], 2)+band
    K = np.asarray(cam.K); v, u = np.nonzero(ok); zz = z[ok]
    P = np.c_[(u-K[0, 2])*zz/K[0, 0], (v-K[1, 2])*zz/K[1, 1], zz]; T = np.asarray(cam.T_base_optical)
    return np.median(P@T[:3, :3].T+T[:3, 3], axis=0), int(len(P))


TAKEOUT_EXTRA_REACH_M, TAKEOUT_EXTRA_LOWER_M = .02, .01  # grip margin for a reversed take-out (meal 2026-11-24: same hand pose, box 4 deg more turned -> closed on air; 2.4 cm deeper / 2 cm lower held)
MIN_GRIP_OVERLAP_M = .03   # reversed place: the finger pads close at least this far below the object's observed top
LOWER_MAX_M = .05          # never lower the hand more than this for it
TOP_BAND_M = .10


def object_top(cam, rows=(.25, .7), cols=(.3, .7)):
    """top (base-frame z, 97th percentile) of the nearest object surface beyond the open fingers in the wrist view"""
    z = np.asarray(cam.depth_m, float); h, w = z.shape; win = np.zeros_like(z, bool)
    win[int(rows[0]*h):int(rows[1]*h), int(cols[0]*w):int(cols[1]*w)] = True
    ok = win & np.isfinite(z) & (z > FINGER_DEPTH_M)
    if ok.sum() < MIN_POINTS:return None, int(ok.sum())
    ok &= z < np.percentile(z[ok], 2)+TOP_BAND_M
    K = np.asarray(cam.K); v, u = np.nonzero(ok); zz = z[ok]
    P = np.c_[(u-K[0, 2])*zz/K[0, 0], (v-K[1, 2])*zz/K[1, 1], zz]; T = np.asarray(cam.T_base_optical)
    return float(np.percentile((P@T[:3, :3].T+T[:3, 3])[:, 2], 97)), int(len(P))


def pads_now(model, data):
    """our finger pads' midpoint in our base frame (arm FK)"""
    b = model.body('robot_0/base').id; Rb = data.xmat[b].reshape(3, 3)
    mid = .5*(data.xpos[model.body('robot_0/gripper/left_pad').id]+data.xpos[model.body('robot_0/gripper/right_pad').id])
    return Rb.T@(mid-data.xpos[b])


def held_centroid(cam):
    """Centroid (wrist optical frame) of the non-robot depth pixels next to the hand, and their count."""
    z = np.asarray(cam.depth_m, float); ok = np.isfinite(z) & (z > .03) & (z < NEAR_M)
    if cam.robot_mask is not None:ok &= ~np.asarray(cam.robot_mask, bool)
    if ok.sum() < MIN_POINTS:return None, int(ok.sum())
    K = np.asarray(cam.K); v, u = np.nonzero(ok); zz = z[ok]
    P = np.c_[(u-K[0, 2])*zz/K[0, 0], (v-K[1, 2])*zz/K[1, 1], zz]
    return np.median(P, axis=0), int(len(P))


class PastRobot:
    """The past robot from its own ACTION LOG only (2026-11-27, user: never read the past episode's scene, object
    poses/shapes or contacts): its recorded commands (commands.npz: base, arm, gripper) and skill segments
    (memory.json), forward kinematics of those commanded arm joints on OUR robot model (a scratch MjData, robot joints
    only), its saved navigation-camera keyframes and its own SLAM poses (visual memory streams). No past scene file."""
    def __init__(self, episode, model):
        import mujoco
        from scripts.test_nav_replay import ACTIONS
        self.ep, self.m = episode, model; self.d = mujoco.MjData(model)
        self.mem = json.loads((ACTIONS/episode/'memory.json').read_text()); names = self.mem['recording']['actuator_names']
        with np.load(ACTIONS/episode/'commands.npz') as z:self.ctime = z['time'].copy(); self.ctrl = z['ctrl'].copy()
        self.gi = names.index('robot_0/gripper/fingers_actuator')
        self.arm = [names.index(f'robot_0/fr3_joint{i}') for i in range(1, 8)]
        self.base = [names.index(f'robot_0/{n}_act') for n in ('base_x', 'base_y', 'base_theta')]
        self.qa = [model.joint(f'robot_0/fr3_joint{i}').qposadr[0] for i in range(1, 8)]
        self.qb = [model.joint(f'robot_0/{n}').qposadr[0] for n in ('base_x', 'base_y', 'base_theta')]
        self.b = model.body('robot_0/base').id; self.g = model.body('robot_0/gripper/base').id
        self.pads = (model.body('robot_0/gripper/left_pad').id, model.body('robot_0/gripper/right_pad').id)
        self.site = model.site('robot_0/gripper/grasp_site').id  # the fingertips' centre (robot geometry)

    def _pose(self, b):
        T = np.eye(4); T[:3, :3] = self.d.xmat[b].reshape(3, 3); T[:3, 3] = self.d.xpos[b]; return T

    def hand(self, k):
        """(T_base_gripper, pads midpoint) in the base frame at command k: FK of the commanded arm joints"""
        import mujoco
        self.d.qpos[self.qb] = 0.; self.d.qpos[self.qa] = self.ctrl[k, self.arm]; mujoco.mj_kinematics(self.m, self.d)
        Tb = self._pose(self.b); Ti = np.linalg.inv(Tb)
        mid = .5*(self.d.xpos[self.pads[0]]+self.d.xpos[self.pads[1]])
        return Ti@self._pose(self.g), (Ti@np.r_[mid, 1.])[:3]

    def tcp(self, k):
        """the fingertips' centre (grasp site) in the base frame at command k: FK of the commanded arm joints"""
        self.hand(k); Ti = np.linalg.inv(self._pose(self.b)); return (Ti@np.r_[self.d.site_xpos[self.site], 1.])[:3]

    def base_rel(self, k0, k):
        """base pose at command k in the base frame at command k0 (the recorded base commands, odometry frame)"""
        from scripts.test_nav_replay import planar
        return np.linalg.inv(planar(*self.ctrl[k0, self.base]))@planar(*self.ctrl[k, self.base])

    def closed_k(self, seg):
        """first command (absolute index) in the segment where the gripper is commanded closed after open"""
        a, b = seg['action_range']; g = self.ctrl[a:b, self.gi] > 127
        idx = np.flatnonzero(np.diff(g.astype(int)) == 1)
        return None if not len(idx) else a+int(idx[0])+1

    def opened_k(self, seg):
        a, b = seg['action_range']; g = self.ctrl[a:b, self.gi] > 127
        idx = np.flatnonzero(np.diff(g.astype(int)) == -1)
        return None if not len(idx) else a+int(idx[0])+1

    def pick_before(self, seg):
        """the past robot's pick segment of the object it carries into place segment `seg` (its last pick before)"""
        picks = [s for s in self.mem['skills'] if s['skill'] == 'pick' and s['action_range'][0] < seg['action_range'][0]]
        return picks[-1] if picks else None

    def pregrasp_hold(self, pick_seg, wrist):
        """where the past object sat in the past gripper frame after its pick, measured like our own in_hand_from_pick:
        the perceived object's bounds centre just before the grasp, in the gripper frame at the close. Here from every
        RECORDED wrist RGB-D frame in the pre-grasp window (object above its support, blob grown from the fingertips'
        point at the close; commanded-joint FK), median over the frames; None when seen in fewer than HOLD_MIN_VIEWS
        frames or the frames disagree by more than HOLD_MAX_SPREAD_M. Then the per-object-type offset between that
        visible-bounds centre and the object's own centre (HOLD_OFFSET_GRIPPER, calibrated offline once)."""
        kc = self.closed_k(pick_seg)
        if kc is None:return None
        t_close = float(self.ctime[kc]); T_c = self.tcp(kc); Tg, _ = self.hand(kc); est = []
        for i in np.flatnonzero((wrist.t >= t_close+HOLD_WINDOW_S[0]-.01) & (wrist.t <= t_close+HOLD_WINDOW_S[1]+.01)):
            cam = wrist.camera(float(wrist.t[i]))
            kf = int(min(np.searchsorted(self.ctime, cam.time), len(self.ctime)-1)); T_f_c = self.base_rel(kf, kc)
            P = _cloud3d(np.asarray(cam.depth_m, float), cam.robot_mask, np.asarray(cam.K), np.asarray(cam.T_base_optical), step=1, near=.05)
            O = object_points(P, (T_f_c@np.r_[T_c, 1.])[:3])
            if O is None:continue
            est.append((np.linalg.inv(T_f_c@Tg)@np.r_[.5*(O.min(0)+O.max(0)), 1.])[:3])
        if len(est) < HOLD_MIN_VIEWS:return None
        E = np.asarray(est); h = np.median(E, axis=0)
        if float(np.median(np.linalg.norm(E-h, axis=1))) > HOLD_MAX_SPREAD_M:return None
        off = HOLD_OFFSET_GRIPPER.get(pick_seg['arguments'].get('target', '').rsplit('_', 1)[0], {})
        h = h-np.asarray(off.get('+' if h[0] >= 0 else '-', (0., 0., 0.)))  # the side of the hand the object is on
        return h if float(np.linalg.norm(h)) < .30 else None


KF_MIN_POINTS = 60
HOLD_WINDOW_S = (-1.2, -.4)   # the recorded pre-grasp wrist frames (memory/wrist_memory, recorded once offline 2026-11-27)
HOLD_MIN_VIEWS, HOLD_MAX_SPREAD_M = 3, .02
# visible-bounds centre minus the object's own centre, gripper frame, per object type: calibrated ONCE offline on the 70
# history episodes on 2026-11-27 (the calibration script is not part of the eval code; held-out half residual medians
# 0.1-2 cm). A fixed constant per
# object type, like a sensor calibration; never computed from the past scene at test time.
HOLD_OFFSET_GRIPPER = {"bowl": {"+": [0.0068, -0.0125, -0.0004], "-": [0.0576, -0.0296, -0.0026]}, "cup": {"+": [-0.0098, -0.0135, 0.0228], "-": [0.0348, -0.0078, 0.0219]}, "drink_bottle": {"+": [0.0006, -0.0134, 0.0263], "-": [-0.0001, -0.0127, 0.0257]}, "food_container": {"+": [-0.0043, -0.0011, 0.0148], "-": [0.0155, 0.015, -0.007]}, "pot": {"+": [0.0772, -0.0349, -0.029], "-": [-0.1045, -0.0097, 0.0049]}, "snack_box": {"+": [-0.0001, -0.0342, -0.0288], "-": [-0.0004, -0.0355, -0.0285]}}  # per object type, per side of the hand (sign of the hold's gripper-x)
OBJ_R_M, OBJ_BELOW_M, OBJ_ABOVE_M, OBJ_LINK_M, OBJ_MAX_EXTENT_M = .18, .30, .12, .015, .40


def object_points(P, grasp):
    """the object at a grasp point in a depth cloud (base frame): the support plane under it (the most common height
    among the points around), the points above that plane, and of those the connected blob (1.5 cm links) that comes
    nearest the grasp point in the plane. Fingers close on a rim / handle / side, so the blob is grown from the grasp,
    not cropped around it. None when nothing is there."""
    from scipy.spatial import cKDTree
    near = P[(np.linalg.norm(P[:, :2]-grasp[:2], axis=1) < OBJ_R_M) & (P[:, 2] > grasp[2]-OBJ_BELOW_M) & (P[:, 2] < grasp[2]+OBJ_ABOVE_M)]
    if len(near) < KF_MIN_POINTS:return None
    h, e = np.histogram(near[:, 2], bins=np.arange(near[:, 2].min(), near[:, 2].max()+.01, .01))
    support = float(e[int(np.argmax(h))]+.005)
    above = near[near[:, 2] > support+.012]
    if len(above) < KF_MIN_POINTS:return None
    tree = cKDTree(above); dxy = np.linalg.norm(above[:, :2]-grasp[:2], axis=1)
    seeds = [int(i) for i in np.argsort(dxy)[:400] if dxy[i] <= .05]
    if not seeds:return None  # nothing within 5 cm of where the fingers closed
    taken = np.zeros(len(above), bool); O = None
    for seed in seeds:  # the nearest point can be a lone speck (finger edge, depth noise): first blob big enough
        if taken[seed]:continue
        seen = np.zeros(len(above), bool); seen[seed] = True; front = [seed]
        while front:
            nb = {j for i in front for j in tree.query_ball_point(above[i], OBJ_LINK_M) if not seen[j]}
            front = list(nb); seen[front] = True
        taken |= seen
        if seen.sum() >= KF_MIN_POINTS:O = above[seen]; break
    if O is None or float(np.ptp(O[:, :2], axis=0).max()) > OBJ_MAX_EXTENT_M:return None  # not one object
    return O


def _cloud3d(depth, mask, K, T_base_opt, step=2, near=.2):
    v = np.isfinite(depth) & (depth > near) & (depth < 3.)
    if mask is not None:v &= ~np.asarray(mask, bool)
    s = np.zeros_like(v); s[::step, ::step] = True; v &= s
    yy, xx = np.nonzero(v); z = depth[yy, xx]
    pc = np.column_stack(((xx-K[0, 2])*z/K[0, 0], (yy-K[1, 2])*z/K[1, 1], z))
    return pc@np.asarray(T_base_opt)[:3, :3].T+np.asarray(T_base_opt)[:3, 3]


def base_cloud(cam):
    """the camera's non-robot depth points in its base frame"""
    z = np.asarray(cam.depth_m, float); ok = np.isfinite(z) & (z > .1)
    if cam.robot_mask is not None:ok &= ~np.asarray(cam.robot_mask, bool)
    K = np.asarray(cam.K); v, u = np.nonzero(ok); zz = z[ok]
    P = np.c_[(u-K[0, 2])*zz/K[0, 0], (v-K[1, 2])*zz/K[1, 1], zz]; T = np.asarray(cam.T_base_optical)
    return P@T[:3, :3].T+T[:3, 3]


def object_offset(now, past, grasp, r=.07, z_lo=-.05, z_hi=.10, iters=4):
    """xy shift of the object at the remembered grasp point: our points minus the past points (both base frame);
    only points above the support (z band around the grasp height) inside a radius that follows the estimate"""
    band = lambda P: P[(P[:, 2] > grasp[2]+z_lo) & (P[:, 2] < grasp[2]+z_hi)]
    now, past = band(now), band(past)
    Pp = past[np.linalg.norm(past[:, :2]-grasp[:2], axis=1) < r]
    if len(Pp) < MIN_POINTS:return None, len(Pp), 0
    cp = np.median(Pp[:, :2], axis=0); dxy = np.zeros(2); n = 0
    for _ in range(iters):
        Pn = now[np.linalg.norm(now[:, :2]-(cp+dxy), axis=1) < r]; n = len(Pn)
        if n < MIN_POINTS:return None, len(Pp), n
        dxy = np.median(Pn[:, :2], axis=0)-cp
    return dxy, len(Pp), n


# <dataset>/memory/wrist_memory/<episode>.h5 (WRIST_MEMORY_ROOT overrides; evaluation/memory_paths.py)
WRIST_FRAME_TOL_S = .06


class RecordedWrist:
    """The past robot's RECORDED wrist RGB-D frames (memory/wrist_memory: recorded once offline, like the navigation
    visual memory). camera(t) returns the recorded frame nearest t (within WRIST_FRAME_TOL_S) or raises LookupError."""
    def __init__(self, episode):
        import h5py
        from ..evaluation.memory_paths import wrist_root
        self.path = wrist_root()/f'{episode}.h5'
        with h5py.File(self.path) as f:self.t = np.asarray(f['time']); self.K = np.asarray(f['K']); self.Tbo = np.asarray(f['T_base_optical'])
        self._c = {}

    def camera(self, t, key='wrist'):
        import h5py
        from types import SimpleNamespace
        if key != 'wrist':raise LookupError('only the wrist camera is recorded here (navigation: the visual memory keyframes)')
        i = int(np.argmin(np.abs(self.t-t)))
        if abs(float(self.t[i])-t) > WRIST_FRAME_TOL_S:raise LookupError(f'no recorded wrist frame within {WRIST_FRAME_TOL_S} s of {t:.2f}')
        if i not in self._c:
            with h5py.File(self.path) as f:
                self._c[i] = SimpleNamespace(rgb=np.asarray(f['rgb'][i]), depth_m=np.asarray(f['depth_m'][i], float), robot_mask=np.asarray(f['robot_mask'][i]),
                                             K=self.K, T_base_optical=self.Tbo[i], time=float(self.t[i]))
        return self._c[i]


def _T_base_gripper(model, data):
    """our gripper in our base frame (our own arm joints, forward kinematics of the live robot)"""
    def T(b):
        M = np.eye(4); M[:3, :3] = data.xmat[b].reshape(3, 3); M[:3, 3] = data.xpos[b]; return M
    return np.linalg.inv(T(model.body('robot_0/base').id))@T(model.body('robot_0/gripper/base').id)


def hold_in_gripper(cam, T_base_gripper):
    """the held object's visible points next to the hand (wrist RGB-D, robot pixels masked), their centroid in the
    gripper frame; (centroid, n) or (None, n)"""
    c, n = held_centroid(cam)
    if c is None:return None, n
    pb = np.asarray(cam.T_base_optical)@np.r_[c, 1.]
    return (np.linalg.inv(T_base_gripper)@pb)[:3], n


class WristObjectMatch:
    """navigator.object_shift hook: base-frame (dx, dy) that moves OUR held object onto the past one at the release.
    Reads only our live sensors/joints and the past robot's recorded data: its commands (PastRobot), its recorded wrist
    RGB-D frames (RecordedWrist) - never the past scene (user decision 2026-11-27)."""
    def __init__(self, runtime, log=None):
        self.rt, self.cache, self.log = runtime, {}, log if log is not None else []

    def _past(self, ep):
        return self.cache.get(('robot', ep)) or self.cache.setdefault(('robot', ep), PastRobot(ep, self.rt.model))

    def _wrist(self, ep):
        return self.cache.get(('wrist', ep)) or self.cache.setdefault(('wrist', ep), RecordedWrist(ep))

    def __call__(self, skill, location, tgt, seg):
        ep = tgt['episode']
        if skill == 'pick':return None  # picks: corrected at the pre-grasp wrist check (pregrasp)
        pr = self._past(ep); ko = pr.opened_k(seg)
        if ko is None or getattr(self, 'in_hand', None) is None:return None
        # our in-hand pose from our own pick vs the past robot's, both measured the same way (perceived object bounds
        # centre before the grasp, in the gripper frame at the close); the past one from its recorded wrist frame
        try:h_past = self.past_hold(dict(episode=ep, segment=seg['segment_id']))
        except LookupError as exc:self.log.append(dict(skill=skill, location=location, result=f'past hold not measured: {exc}'[:120]));return None
        Rbg = pr.hand(ko)[0][:3, :3]  # the past hand at its release (its commanded joints)
        h_now = np.asarray(self.in_hand, float); self.flip = False
        if h_now[0]*h_past[0] < 0 and min(abs(h_now[0]), abs(h_past[0])) > FLIP_MIN_X_M and location in FLIP_LOCATIONS:
            # round object (bowl/cup) gripped on the other side of its rim: a 180 deg turn about the hand axis makes
            # our hold match the past one; the replay turns the wrist by the same amount
            h_now = h_now*np.array([-1., -1., 1.]); self.flip = True
        d3 = Rbg@(h_past-h_now); dxy = d3[:2]
        # our object at the release = past object - d3: d3_z > 0 means ours is LOWER (hits the shelf/rim);
        # higher is harmless (it drops a few cm: the drink into the microwave, isolated runs)
        if float(d3[2]) > PLACE_MAX_DZ_M and location in PLACE_DZ_GATED:
            self.log.append(dict(skill=skill, location=location, result='our object would be released lower than the past one', dz_m=round(float(d3[2]), 3)))
            return np.array([9., 9.])  # beyond OBJECT_SHIFT_MAX: memory refuses this place
        self.log.append(dict(skill=skill, location=location, memory=f"{ep}/{seg.get('segment_id')}", result='measured from our pick', wrist_flip=self.flip,
                             in_hand_now_m=np.round(self.in_hand, 3).tolist(), in_hand_past_m=np.round(h_past, 3).tolist(), shift_m=np.round(dxy, 3).tolist()))
        return dxy

    def past_hold(self, row):
        """the past robot's hold of the object it carries into remembered place `row` (episode, segment): measured at
        its pick before that place (pregrasp_hold), cached; LookupError when not measurable"""
        key = ('hold', row['episode'], row['segment'])
        if key not in self.cache:
            pr = self._past(row['episode']); seg = next(x for x in pr.mem['skills'] if x['segment_id'] == row['segment'])
            pk = seg if seg['skill'] == 'pick' else pr.pick_before(seg)
            h = None if pk is None else pr.pregrasp_hold(pk, self._wrist(row['episode']))
            self.cache[key] = h
        if self.cache[key] is None:raise LookupError('past held object not seen in its recorded pre-grasp wrist frame')
        return self.cache[key]

    def press(self, tgt, seg, button):
        return PressCheck(self.rt, tgt, seg, button, self.log, self._past(tgt['episode']), self._wrist(tgt['episode']))

    def pregrasp(self, tgt, seg, reverse=False):
        """replay hook for a remembered pick: at recorded command k (just before the fingers close) compare our wrist
        view with the past robot's RECORDED wrist view at the same command; returns the base-frame (dx, dy) that puts
        our hand where the object is relative to the hand in the past (None = not measured)"""
        ep = tgt['episode']; pr = self._past(ep); wrist = self._wrist(ep); ct = pr.ctime
        a, b = seg['action_range']

        def hook(k):
            if reverse:k = (b-a)-1-k  # reversed replay: command k is the past segment's command from the end
            cam_now = self.rt.sensors.capture(camera_keys=('wrist',)).cameras['wrist']
            p_now, n_now = nearest_object(cam_now)
            try:p_past, n_past = nearest_object(wrist.camera(float(ct[a+k])))
            except LookupError as exc:p_past, n_past = None, str(exc)[:80]
            row = dict(skill='pregrasp', memory=f"{ep}/{seg.get('segment_id')}", points_now=n_now, points_past=n_past)
            if p_now is None or p_past is None:row['result'] = 'object not seen by the wrist camera'; self.log.append(row); return None
            dxy = (p_now-p_past)[:2]  # both in their own base frames: the object moved this much relative to the hand path
            row.update(result='measured', object_now_base_m=np.round(p_now, 3).tolist(), object_past_base_m=np.round(p_past, 3).tolist(), shift_m=np.round(dxy, 3).tolist())
            self.log.append(row); return dxy
        return hook


def in_hand_from_pick(pick_dir, stage_log):
    """Where OUR held object sits in the hand, from our own pick: the object's perceived surface (the pick stage's
    OBSERVED_WORLD target bounds, robot base frame, just before the grasp) expressed in the gripper frame at the moment
    the fingers closed (the pick's own recording: arm joints -> forward kinematics; the base stands still during the
    pick). Returns the object centre in the gripper frame, or None."""
    import ast, h5py, mujoco
    line = None
    for l in Path(stage_log).read_text(errors='replace').splitlines():
        if 'OBSERVED_WORLD ' in l:line = l
    if line is None:return None
    world = ast.literal_eval(line[line.index('{'):])
    lo, hi = (np.asarray(b, float) for b in world['target_surface_bounds']); c = .5*(lo+hi)
    rec = Path(pick_dir)/'physics'/'state_recording'
    m = mujoco.MjModel.from_binary_path(str(rec/'model.mjb')); d = mujoco.MjData(m)
    gi = m.actuator('robot_0/gripper/fingers_actuator').id
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    with h5py.File(rec/'states.h5') as f:
        S = f['states']; was_open = False
        for i in range(len(S)):
            mujoco.mj_setState(m, d, np.asarray(S[i]), spec)
            if d.ctrl[gi] <= 127:was_open = True
            elif was_open:break  # the grasp: open -> closed (the travel tuck starts closed; store-run 5 took that frame)
        else:return None
    mujoco.mj_kinematics(m, d)
    def T(b):
        M = np.eye(4); M[:3, :3] = d.xmat[b].reshape(3, 3); M[:3, 3] = d.xpos[b]; return M
    T_bg = np.linalg.inv(T(m.body('robot_0/base').id))@T(m.body('robot_0/gripper/base').id)
    h = (np.linalg.inv(T_bg)@np.r_[c, 1.])[:3]
    return h if float(np.linalg.norm(h)) < .30 else None  # an object 30+ cm from the hand is not a hold estimate


def press_touch_k(pr, seg):
    """absolute command index of the past arm's farthest reach in a press segment (its own commanded joints, FK)
    = the touch (offline check against the recordings: within 0.15 s of the first finger-button contact)"""
    a, b = seg['action_range']; P0 = pr.hand(a)[1]; best = (-1., a)
    for k in range(a, b, 25):
        p = (pr.base_rel(a, k)@np.r_[pr.hand(k)[1], 1.])[:3]; d = float(np.linalg.norm(p-P0))
        if d > best[0]:best = (d, k)
    return best[1]


PRESS_REACH_TO_TOUCH_K = 55  # commands (2 ms): 0.11 s, offline check against the recordings 2026-11-27
PRESS_EXTRA_M = .008  # extra press depth along the wrist camera's view direction (base plane)


class PressCheck:
    """Replay hook for a remembered button press: just before the past robot's touch, compare our wrist view with the
    past robot's at the same command: panel depth difference (and the image shift when the match is clear) -> base
    shift, so the fingertip reaches the button (a press is a few mm deep: meal run 5 stopped 0.5 cm short)."""
    max_m = .04

    def __init__(self, runtime, tgt, seg, button, log, past_robot, wrist):
        self.rt, self.log = runtime, log; self.ep = tgt['episode']; self.seg = seg
        self.wrist = wrist; self.a = seg['action_range'][0]; self.ct = past_robot.ctime
        # from the segment start (the replay's command index); the farthest commanded reach comes 0.11 s (max 0.13 s, all
        # 28 recorded presses) before the finger meets the button: the arm still closes the last millimetres
        self.event_k = press_touch_k(past_robot, seg)-self.a+PRESS_REACH_TO_TOUCH_K

    def __call__(self, k):
        import cv2
        now = self.rt.sensors.capture(camera_keys=('wrist',)).cameras['wrist']; past = self.wrist.camera(float(self.ct[self.a+k]))
        h, w = now.depth_m.shape; crop = lambda x: np.asarray(x)[int(.05*h):int(.6*h), int(.15*w):int(.85*w)]
        zn, zp = float(np.nanmedian(crop(now.depth_m))), float(np.nanmedian(crop(past.depth_m)))
        gray = lambda c: cv2.cvtColor(np.ascontiguousarray(crop(c.rgb)), cv2.COLOR_RGB2GRAY).astype(np.float32)
        (du, dv), resp = cv2.phaseCorrelate(gray(past), gray(now)); K = np.asarray(now.K)
        lateral = resp >= .3
        off = np.array([du*zn/K[0, 0] if lateral else 0., dv*zn/K[1, 1] if lateral else 0., zn-zp])
        dxy = (np.asarray(now.T_base_optical)[:3, :3]@off)[:2]
        # press a little deeper than the past robot did: its press is at the edge (meal 2026-11-24: the Start finger stopped
        # 1 mm short of contact, the microwave never started); the panel stops the finger, a button needs travel
        view = np.asarray(now.T_base_optical)[:3, 2][:2]; view = view/max(1e-9, float(np.linalg.norm(view)))
        dxy = dxy+PRESS_EXTRA_M*view
        self.log.append(dict(skill='press_check', memory=f"{self.ep}/{self.seg.get('segment_id')}", panel_depth_now_m=round(zn, 4),
                             panel_depth_past_m=round(zp, 4), image_shift_px=[round(du, 1), round(dv, 1)], match=round(float(resp), 3),
                             lateral_used=bool(lateral), shift_m=np.round(dxy, 4).tolist()))
        return dxy


LOOK_AFTER_RELEASE_S = .8  # where both looks happen: this long after the remembered release (hand withdrawing)


def _release_k(seg, gi):
    from scripts.test_nav_replay import ACTIONS
    return None


class ReleaseLook:
    """Place replay hook: just after the release, where OUR wrist camera sees the object we let go (base frame)."""
    def __init__(self, runtime, seg, ctrl_gripper, dt, store):
        a, b = seg['action_range']; g = ctrl_gripper[a:b] > 127
        op = np.flatnonzero(np.diff(g.astype(int)) == -1)
        self.rt, self.store, self.n = runtime, store, b-a
        self.k_rel = None if not len(op) else int(op[0])+1
        self.check_k = None if self.k_rel is None else min(self.n-1, self.k_rel+int(LOOK_AFTER_RELEASE_S/dt))
        store['look_k'] = self.check_k
        self.pre_look_k = None if self.k_rel is None else max(0, self.k_rel-int(.3/dt))  # still holding: the object's top in the hand

    def pre_look(self, k):
        cam = self.rt.sensors.capture(camera_keys=('wrist',)).cameras['wrist']
        top, n = object_top(cam); self.store['held_top'] = top; self.store['held_top_points'] = n
        p, _ = nearest_object(cam); self.store['held_obj'] = None if p is None else np.asarray(p).tolist()  # front while still held

    def __call__(self, k):
        cam = self.rt.sensors.capture(camera_keys=('wrist',)).cameras['wrist']
        p, n = nearest_object(cam); self.store['release_obj'] = None if p is None else np.asarray(p).tolist(); self.store['release_points'] = n
        self.store['release_top'] = object_top(cam)[0]
        return None  # a look only, no shift


class ReverseLook:
    """Reversed-place hook: at the same command as the release look, our object now vs then -> base shift."""
    max_m = .09  # its push-in is clipped to 8 cm; the replay's 6 cm pre-grasp abort refused a 7.7 cm correction (2026-11-24)

    def __init__(self, runtime, lp, log):
        self.rt, self.lp, self.log = runtime, lp, log; n = lp.get('n_cmd')
        self.check_k = None if (lp.get('look_k') is None or n is None) else (n-1)-int(lp['look_k'])

    def __call__(self, k):
        if self.lp.get('release_obj') is None:return None
        cam = self.rt.sensors.capture(camera_keys=('wrist',)).cameras['wrist']; p, n = nearest_object(cam)
        row = dict(skill='reverse_look', points_now=n)
        if p is None:row['result'] = 'object not seen'; self.log.append(row); return None
        raw = (np.asarray(p)-np.asarray(self.lp['release_obj']))[:2]
        # plus how far the object slid deeper when our place let go (its front just before vs just after the release;
        # meal 2026-11-24: a 1.5 cm slide left the reversed close in front of the box)
        held = self.lp.get('held_obj')
        slide = float(np.asarray(self.lp['release_obj'])[0]-np.asarray(held)[0]) if held is not None else 0.
        row_slide = round(slide, 3); raw = raw+np.array([max(0., slide), 0.])
        # only the reach direction (base x), and only deeper: a released object slides/settles further in (meal run 12:
        # the food box slid 3.7 cm deeper and the reversed close met air); the sideways part of this view is unreliable
        dxy = np.array([float(np.clip(raw[0]+TAKEOUT_EXTRA_REACH_M, 0., .08)), 0.])
        row['raw_shift_m'] = np.round(raw, 3).tolist(); row['release_slide_m'] = row_slide
        row.update(result='measured', object_now_m=np.round(p, 3).tolist(), at_release_m=np.round(self.lp['release_obj'], 3).tolist(), shift_m=np.round(dxy, 3).tolist())
        self.log.append(row); print('REVERSE_LOOK', json.dumps(row, default=str), flush=True); return dxy

    def before_close(self, model, data):
        """right before the reversed close: when our place let go, the object dropped/settled (meal 2026-11-23: the food
        box fell ~2.5 cm; the reversed close at the release height met air above it). Our place measured the object's
        top in the hand just before the release and just after it; lower the hand by that drop plus any settling seen
        now, so the fingers come back to the height that held it. Returns the world dz (<= 0)."""
        held, rel = self.lp.get('held_top'), self.lp.get('release_top')
        cam = self.rt.sensors.capture(camera_keys=('wrist',)).cameras['wrist']; top, n = object_top(cam)
        row = dict(skill='reverse_close_height', points=n, held_top_m=held, release_top_m=rel, top_now_m=None if top is None else round(top, 3))
        if held is None or rel is None or top is None:row['result'] = 'no measurement'; self.log.append(row); print('REVERSE_CLOSE_HEIGHT', json.dumps(row), flush=True); return 0.
        lower = float(np.clip((held-rel)+(rel-top)+TAKEOUT_EXTRA_LOWER_M, 0., LOWER_MAX_M))
        row.update(result='measured', drop_m=round(held-rel, 3), settle_m=round(rel-top, 3), lower_m=round(lower, 3)); self.log.append(row)
        print('REVERSE_CLOSE_HEIGHT', json.dumps(row), flush=True)
        return -lower
