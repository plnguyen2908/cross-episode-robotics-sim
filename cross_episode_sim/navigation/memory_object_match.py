"""Where a held object sits in the hand, from the wrist camera: our live view vs the past robot's view.

A remembered place releases the object where the past robot's hand released ITS object. When our object sits
differently in the hand, the base shifts by that difference first (memory_navigation.shift_base) so the object -
not the hand - lands where the past object landed.

Live side: the onboard wrist camera (depth + the robot-geometry mask from encoder FK). Past side: the past robot's
wrist view at the same moment of its segment, re-rendered from its own recorded state through the same onboard
sensor model (the history streams kept only the navigation camera). No simulator truth of the current episode is used.
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


class PastWrist:
    """The past episode's robot at a recorded time, seen through the onboard sensor model."""
    def __init__(self, episode):
        import mujoco
        from scripts.test_location_matrix import PLAN
        src = Path(PLAN[episode]['source'])/'action_recording'
        self.m = mujoco.MjModel.from_binary_path(str(src/'model.mjb')); self.d = mujoco.MjData(self.m)
        a = np.load(src/'actions.npz'); self.t, self.q = a['time'], a['qpos']

    def camera(self, t, key='wrist'):
        import mujoco
        from cross_episode_sim.sensors.onboard import OnboardSensorInterface
        self.d.qpos[:] = self.q[min(np.searchsorted(self.t, t), len(self.t)-1)]; mujoco.mj_forward(self.m, self.d)
        with OnboardSensorInterface(self.m, self.d) as s:return s.capture(camera_keys=(key,)).cameras[key]

    def pads_in_base(self, t):
        """the past fingers' midpoint at time t in its base frame (its own arm FK)"""
        import mujoco
        self.d.qpos[:] = self.q[min(np.searchsorted(self.t, t), len(self.t)-1)]; mujoco.mj_kinematics(self.m, self.d)
        b = self.m.body('robot_0/base').id; Rb = self.d.xmat[b].reshape(3, 3)
        mid = .5*(self.d.xpos[self.m.body('robot_0/gripper/left_pad').id]+self.d.xpos[self.m.body('robot_0/gripper/right_pad').id])
        return Rb.T@(mid-self.d.xpos[b])


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


class WristObjectMatch:
    """navigator.object_shift hook: base-frame (dx, dy) that moves OUR held object onto the past one at the release."""
    def __init__(self, runtime, log=None):
        self.rt, self.cache, self.log = runtime, {}, log if log is not None else []

    def __call__(self, skill, location, tgt, seg):
        from scripts.test_nav_replay import ACTIONS
        ep = tgt['episode']; past = self.cache.get(ep) or self.cache.setdefault(ep, PastWrist(ep))
        mem = json.loads((ACTIONS/ep/'memory.json').read_text())
        gi = mem['recording']['actuator_names'].index('robot_0/gripper/fingers_actuator')
        with np.load(ACTIONS/ep/'commands.npz') as z:ct = z['time']; g = z['ctrl'][:, gi]
        a, b = seg['action_range']
        if skill == 'pick' and not PICK_NAV_VIEW:return None  # picks: corrected at the pre-grasp wrist check instead
        if skill == 'pick':
            # the object where the past fingers closed: its points around that spot in the past navigation view (at
            # the segment start, before the arm moved) vs in our live navigation view from the aligned pose
            closed = np.flatnonzero(np.diff((g[a:b] > 127).astype(int)) == 1)
            if not len(closed):return None
            t_start, t_grasp = float(ct[a]), float(ct[a+int(closed[0])+1])
            grasp = past.pads_in_base(t_grasp)
            now = base_cloud(self.rt.sensors.capture(camera_keys=('navigation',)).cameras['navigation'])
            then = base_cloud(past.camera(t_start+.1, 'navigation'))
            dxy, n_past, n_now = object_offset(now, then, grasp)
            row = dict(skill=skill, location=location, memory=f"{ep}/{seg.get('segment_id')}", grasp_base_m=np.round(grasp, 3).tolist(),
                       points_past=n_past, points_now=n_now, result='measured' if dxy is not None else 'object not seen at the remembered grasp')
            if dxy is not None:row['shift_m'] = np.round(dxy, 3).tolist()
            self.log.append(row); return dxy
        opened = np.flatnonzero(np.diff((g[a:b] > 127).astype(int)) == -1)
        if getattr(self, 'in_hand', None) is not None and len(opened):
            # our in-hand pose from our own pick vs the past robot's object in its hand at the start of this place
            from scripts.test_constrained_replay import PastEpisode, body_of, pose
            pe = self.cache.get(('rec', ep)) or self.cache.setdefault(('rec', ep), PastEpisode(ep))
            key = pe.held_key(seg); pd = pe.at(float(ct[a])+.1)
            Tg = pose(pd, pe.m.body('robot_0/gripper/base').id); h_past = (np.linalg.inv(Tg)@np.r_[pd.xpos[body_of(pe.m, key)], 1.])[:3]
            pd = pe.at(float(ct[a+int(opened[0])+1])-.05)
            Rbg = (np.linalg.inv(pose(pd, pe.m.body('robot_0/base').id))@pose(pd, pe.m.body('robot_0/gripper/base').id))[:3, :3]
            h_now = np.asarray(self.in_hand, float); self.flip = False
            if h_now[0]*h_past[0] < 0 and min(abs(h_now[0]), abs(h_past[0])) > FLIP_MIN_X_M and location in FLIP_LOCATIONS:
                # round object (bowl/cup) gripped on the other side of its rim: a 180 deg turn about the hand axis makes
                # our hold match the past one; the replay turns the wrist by the same amount
                h_now = h_now*np.array([-1., -1., 1.]); self.flip = True
            d3 = Rbg@(h_past-h_now); dxy = d3[:2]
            # our object at the release = past object - d3: d3_z > 0 means ours is LOWER (hits the shelf/rim);
            # higher is harmless (it drops a few cm: the drink into the microwave, isolated runs)
            if float(d3[2]) > PLACE_MAX_DZ_M and location in PLACE_DZ_GATED:
                # the hold differs vertically at the release: a base shift cannot fix that (store run 6: bowl 2 released
                # 8 cm below the shelf, hit its edge and fell out) -> refuse, the baseline place follows
                self.log.append(dict(skill=skill, location=location, result='our object would be released lower than the past one', dz_m=round(float(d3[2]), 3)))
                return np.array([9., 9.])  # beyond OBJECT_SHIFT_MAX: memory refuses this place
            self.log.append(dict(skill=skill, location=location, memory=f"{ep}/{seg.get('segment_id')}", result='measured from our pick', wrist_flip=self.flip,
                                 in_hand_now_m=np.round(self.in_hand, 3).tolist(), in_hand_past_m=np.round(h_past, 3).tolist(), shift_m=np.round(dxy, 3).tolist()))
            return dxy
        if not len(opened):return None
        t_start, t_release = float(ct[a]), float(ct[a+int(opened[0])+1])
        cam_now = self.rt.sensors.capture(camera_keys=('wrist',)).cameras['wrist']
        h_now, n_now = held_centroid(cam_now)
        cam_past = past.camera(t_start+.1); h_past, n_past = held_centroid(cam_past)
        row = dict(skill=skill, location=location, memory=f"{ep}/{seg.get('segment_id')}", points_now=n_now, points_past=n_past)
        if h_now is None or h_past is None:row['result'] = 'held object not seen'; self.log.append(row); return None
        R_base_hand = np.asarray(past.camera(t_release-.05).T_base_optical)[:3, :3]  # the past hand at its release
        dxy = (R_base_hand@(h_past-h_now))[:2]
        row.update(result='measured' if PLACE_SHIFT else 'measured (logged only)', in_hand_now_m=np.round(h_now, 3).tolist(), in_hand_past_m=np.round(h_past, 3).tolist(),
                   shift_m=np.round(dxy, 3).tolist()); self.log.append(row)
        return dxy if PLACE_SHIFT else None

    def past_hold(self, row):
        """Where the past robot's object sat in its hand at the start of remembered place `row` (its recording)."""
        key = (row['episode'], row['segment'])
        if key not in self.cache:
            from scripts.test_constrained_replay import PastEpisode, body_of, pose
            pe = self.cache.get(('rec', row['episode'])) or self.cache.setdefault(('rec', row['episode']), PastEpisode(row['episode']))
            seg = next(x for x in pe.mem['skills'] if x['segment_id'] == row['segment'])
            pd = pe.at(float(pe.ctime[seg['action_range'][0]])+.1); k = pe.held_key(seg)
            self.cache[key] = (np.linalg.inv(pose(pd, pe.m.body('robot_0/gripper/base').id))@np.r_[pd.xpos[body_of(pe.m, k)], 1.])[:3]
        return self.cache[key]

    def press(self, tgt, seg, button):
        return PressCheck(self.rt, tgt, seg, button, self.log)

    def pregrasp(self, tgt, seg, reverse=False):
        """replay hook for a remembered pick: at recorded command k (just before the fingers close) compare our wrist
        view with the past robot's at the same command; returns the base-frame (dx, dy) that puts our hand where the
        object is relative to the hand in the past (None = not measured)"""
        from scripts.test_nav_replay import ACTIONS
        ep = tgt['episode']; past = self.cache.get(ep) or self.cache.setdefault(ep, PastWrist(ep))
        with np.load(ACTIONS/ep/'commands.npz') as z:ct = z['time'].copy()
        a, b = seg['action_range']

        def hook(k):
            if reverse:k = (b-a)-1-k  # reversed replay: command k is the past segment's command from the end
            cam_now = self.rt.sensors.capture(camera_keys=('wrist',)).cameras['wrist']
            p_now, n_now = nearest_object(cam_now); p_past, n_past = nearest_object(past.camera(float(ct[a+k])))
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


def press_event_k(past, seg, button):
    """Command index (from the segment start) where the past robot's gripper first touched `button` (its own recording)."""
    import mujoco
    a, b = seg['action_range']; gid = past.m.geom(f'microwave_main_group_{button}_button').id
    grip = {i for i in range(past.m.nbody) if past.m.body(i).name.startswith('robot_0/gripper/')}
    for k in range(0, b-a, 25):
        d = past.at(float(past.ctime[a+k])); mujoco.mj_forward(past.m, d)
        for c in d.contact[:d.ncon]:
            if gid in (c.geom1, c.geom2) and past.m.geom_bodyid[c.geom2 if c.geom1 == gid else c.geom1] in grip:return k
    return None


PRESS_EXTRA_M = .008  # extra press depth along the wrist camera's view direction (base plane)


class PressCheck:
    """Replay hook for a remembered button press: just before the past robot's touch, compare our wrist view with the
    past robot's at the same command: panel depth difference (and the image shift when the match is clear) -> base
    shift, so the fingertip reaches the button (a press is a few mm deep: meal run 5 stopped 0.5 cm short)."""
    max_m = .04

    def __init__(self, runtime, tgt, seg, button, log):
        from scripts.test_constrained_replay import PastEpisode
        from scripts.test_nav_replay import ACTIONS
        self.rt, self.log = runtime, log; self.ep = tgt['episode']; self.seg = seg
        self.past = PastEpisode(self.ep); self.wrist = PastWrist(self.ep); self.a = seg['action_range'][0]
        with np.load(ACTIONS/self.ep/'commands.npz') as z:self.ct = z['time'].copy()
        self.event_k = press_event_k(self.past, seg, button)

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
