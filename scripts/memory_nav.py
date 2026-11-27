"""Fast memory navigation: pick the quickest way to a general location from the offline memory index.

Options, all costed by planning on the tour map (no trial driving):
  - join a remembered route at its point nearest the robot (A* join), then replay the rest of it
  - plain A* to the location
Estimated robot time = A* length / JOIN_SPEED + remaining recorded route time. The quickest is executed;
a replay that deviates > ABORT_M from its map path stops and A* finishes. Ends when the robot is within
NEAR of a confident remembered manipulation start at that location (the manipulation replay docks itself).
"""
import json, math, time
from pathlib import Path

import mujoco
import numpy as np

from scripts.test_nav_replay import ACTIONS, planar, flat, encoders, base_cmd_ids

JOIN_SPEED, ASTAR_SPEED = .65, .65   # m/s planning estimate for drive_to (pure pursuit, 0.8 m/s cap)
ABORT_M, NEAR = .25, .30
STEP = .1                            # index route sample spacing (s)
EASE_S = 1.5                         # ease-in when a replay starts from rest
AGREE_M, AGREE_DEG = .30, 15.          # consensus between remembered manipulation starts
SLIDE_M = .35                        # a route this close is joined by a short holonomic lead-in, no join drive
NEAR_GOAL_DIRECT_M = .8  # closer than this to the remembered start: direct A* move only, no route replay
LEAF_SWEEP_M = .42                   # base centre to an opened leaf along a replayed route (stowed arm / mast sweep)


import os
DEBUG = bool(os.environ.get('MEMORY_NAV_DEBUG'))
V_MAX, W_MAX = .8, 1.2             # replay speed limits (m/s, rad/s): the speeds at which onboard SLAM tracked the tours


def playback_index(cmd, dt, n_ease=0):
    """Fractional command index per physics step: the recorded path, retimed so the commanded base
    speed stays <= V_MAX / W_MAX, with a 0 -> 1 rate ramp over the first n_ease steps."""
    v = np.r_[np.linalg.norm(np.diff(cmd[:, :2], axis=0), axis=1)/dt, 0.]
    w = np.r_[np.abs(np.diff(np.unwrap(cmd[:, 2])))/dt, 0.]
    rate = np.minimum(1., np.minimum(V_MAX/np.maximum(v, 1e-9), W_MAX/np.maximum(w, 1e-9)))
    out = []; f = 0.; step = 0
    while f < len(cmd)-1:
        out.append(f); ramp = min(1., (step+1)/n_ease) if n_ease else 1.
        f += ramp*rate[int(f)]; step += 1
    return np.array(out)


def route_seconds(q, j):
    """Replay time from route sample j at the speed limits (samples are STEP s apart as recorded)."""
    q = np.asarray(q)[j:]
    if len(q) < 2:return 0.
    d = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=1); a = np.abs(np.diff(np.unwrap(q[:, 2])))
    return float(np.maximum(STEP, np.maximum(d/V_MAX, a/W_MAX)).sum())+EASE_S/2


FACE_REMEMBERED = False  # on arrival, turn to the remembered start's heading (off: user 2026-11-23, keep run-8 behaviour; only picking was asked)
RECOVER_TURN = True   # turn in place to regain lost SLAM tracking (see recover_tracking)
RECOVER_W = .35      # rad/s


class MemoryNav:
    def __init__(self, index_path, R, model, data, nav):
        self.idx = json.loads(Path(index_path).read_text()); self.R, self.m, self.d, self.nav = R, model, data, nav
        self.res = nav.res

    def pose(self, wait_s=3., recover=True):
        """SLAM pose on the map; stand still up to wait_s for tracking to return, then turn in place to find it again."""
        end = float(self.d.time)+wait_s
        while True:
            P = self.nav.pose()
            if P is not None:
                self._good_th = float(self.d.joint('robot_0/base_theta').qpos[0]); return P
            if float(self.d.time) >= end:break
            self.R.tick(0., 0.)
        return self.recover_tracking() if recover and RECOVER_TURN else None

    def recover_tracking(self):
        """Tracking lost (grocery 000032 2026-11-23: A* turned the robot to face a blank wall corner, SIFT had nothing to
        match and the episode stayed lost): turn slowly in place on the base encoders, first back to the heading of the
        last tracked pose, then sweep, until the keyframe tracker relocks on features it has seen."""
        from cross_episode_sim.control.onboard_base import OnboardBaseActuator
        ids = base_cmd_ids(self.m); dt = self.m.opt.timestep; t0 = float(self.d.time)
        cur = np.array([self.d.joint(f'robot_0/{n}').qpos[0] for n in ('base_x', 'base_y', 'base_theta')])
        last = getattr(self, '_good_th', None)
        goals = ([last] if last is not None else [])+[cur[2]+math.radians(a) for a in (45, -45, 90, -90, 135, -135, 180)]
        P = None; th = cur[2]
        for g in goals:
            n = max(1, int(abs(g-th)/RECOVER_W/dt))
            for k in range(n):
                self.d.ctrl[ids] = [cur[0], cur[1], th+(g-th)*(k+1)/n]; mujoco.mj_step(self.m, self.d)
                if (k+1) % 50 == 0:
                    self.R.sense(); P = self.nav.pose()
                    if P is not None:break
            th = float(self.d.ctrl[ids][2])
            if P is None:
                for k in range(150):
                    self.d.ctrl[ids] = [cur[0], cur[1], th]; mujoco.mj_step(self.m, self.d)
                    if (k+1) % 50 == 0:
                        self.R.sense(); P = self.nav.pose()
                        if P is not None:break
            if P is not None:break
        self.R.base = OnboardBaseActuator(self.m, self.d, motion_guard=self.R.guard)
        for _ in range(25):self.R.tick(0., 0.)
        P = self.nav.pose() if P is not None else None
        print('TRACKING_RECOVERY', json.dumps(dict(recovered=P is not None, turned_deg=round(math.degrees(th-cur[2]), 1),
              robot_s=round(float(self.d.time)-t0, 1))), flush=True)
        if P is not None:self._good_th = float(self.d.joint('robot_0/base_theta').qpos[0])
        return P

    def _astar_len(self, a_xy, b_xy):
        p = self.nav.astar(self.nav.free, self.nav.snap(self.nav.cell(a_xy)), self.nav.snap(self.nav.cell(b_xy)))
        if p is None:return None
        p = np.array(p, float); return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()*self.res)

    def manip_starts(self, location, skills, exclude=None, kind=None):
        """Confident remembered manipulation starts at `location`; when several exist, only those another start
        agrees with (within AGREE_M / AGREE_DEG), so a lone drifted pose cannot be chosen. With `kind` (e.g. 'drink'),
        only the past picks of that kind of object, when there are any (grocery 2026-11-23: edge-first over ALL past
        dining-table picks parked in front of a snack box and the drink pick took the snack): see go()."""
        rows = [r for r in self.idx['segments'] if r['skill'] in skills and r['location'] == location and r['start_confident']
                and r['episode'] != exclude]
        af = getattr(self, 'arg_filter', None)
        if af:
            # requested call arguments (a knob turn's on/off state): only matching past segments, never a fallback
            rows = [r for r in rows if all((r.get('arguments') or {}).get(k) == v for k, v in af.items())]
            if not rows:return []
        def agree(a, b):
            dy = math.degrees(abs(math.atan2(math.sin(a['start'][2]-b['start'][2]), math.cos(a['start'][2]-b['start'][2]))))
            return np.linalg.norm(np.subtract(a['start'][:2], b['start'][:2])) <= AGREE_M and dy <= AGREE_DEG
        backed = [r for r in rows if any(o is not r and o['episode'] != r['episode'] and agree(r, o) for o in rows)]
        if backed or rows:return backed or rows
        return self.chained_starts(location, skills, exclude)

    def chained_starts(self, location, skills, exclude=None):
        """No confident start here (e.g. in front of an open fridge door): place the segment through a confident
        segment of the SAME past episode - its recorded base commands share one odometry frame, so the confident
        segment's command->map transform also places this segment's first command."""
        out = []
        for r in self.idx['segments']:
            if r['skill'] not in skills or r['location'] != location or r['episode'] == exclude:continue
            anchors = [a for a in self.idx['segments'] if a['episode'] == r['episode'] and a['start_confident']]
            if not anchors:continue
            a = min(anchors, key=lambda x: abs(x['action_range'][0]-r['action_range'][0]))  # nearest in time: least drift
            names = json.loads((ACTIONS/r['episode']/'memory.json').read_text())['recording']['actuator_names']
            bi = [names.index(f'robot_0/{n}') for n in ('base_x_act', 'base_y_act', 'base_theta_act')]
            with np.load(ACTIONS/r['episode']/'commands.npz') as z:c = z['ctrl'][r['action_range'][0], bi]
            T = np.asarray(a['T_map_rec'])@planar(*c)
            out.append(dict(r, T_map_rec=a['T_map_rec'], start=[float(T[0, 3]), float(T[1, 3]), float(math.atan2(T[1, 0], T[0, 0]))],
                            chained_from=f"{a['segment']} ({abs(a['action_range'][0]-r['action_range'][0])*.002:.0f} s apart)"))
        return out

    def plan(self, location, goal_xy, exclude=None):
        """All options with estimated robot seconds; cheapest first."""
        P = self.pose()
        if P is None:return []
        here = P[:2, 3]; opts = []
        direct = self._astar_len(here, goal_xy)
        if direct is not None:opts.append(dict(kind='astar', seconds=direct/ASTAR_SPEED, astar_m=round(direct, 2)))
        routes = [r for r in self.idx['segments'] if r['skill'] == 'navigate' and r['location'] == location
                  and r['start_confident'] and r['episode'] != exclude]
        pre = []
        for r in routes:
            q = np.asarray(r['route']); n = len(q); last = max(1, int(n*.8))
            d = np.linalg.norm(q[:last, :2]-here, axis=1)
            end_gap = float(np.linalg.norm(q[-1, :2]-goal_xy))
            for j in np.argsort(d)[:3]:
                pre.append((float(d[j])/JOIN_SPEED+route_seconds(q, j)+end_gap/ASTAR_SPEED, r, int(j), end_gap))
        pre.sort(key=lambda x: x[0])
        for est, r, j, end_gap in pre[:6]:
            q = np.asarray(r['route']); L = self._astar_len(here, q[j, :2])
            if L is None:continue
            opts.append(dict(kind='replay', seconds=L/JOIN_SPEED+route_seconds(q, j)+end_gap/ASTAR_SPEED, join_m=round(L, 2),
                             replay_s=round(route_seconds(q, j), 1), memory=f"{r['episode']}/{r['segment']}", row=r, j=j))
        return sorted(opts, key=lambda o: o['seconds'])

    def replay_from(self, r, j):
        names = json.loads((ACTIONS/r['episode']/'memory.json').read_text())['recording']['actuator_names']
        from cross_episode_sim.mapping.action_memory import rebase_planar_commands
        from cross_episode_sim.control.onboard_base import OnboardBaseActuator
        a, b = r['action_range']; k0 = a+int(r['route'][j][3])
        with np.load(ACTIONS/r['episode']/'commands.npz') as z:ctrl = z['ctrl'][k0:b].copy()
        bi = [names.index(f'robot_0/{n}') for n in ('base_x_act', 'base_y_act', 'base_theta_act')]
        T_map_rec = np.asarray(r['T_map_rec']); P = self.pose()
        if P is None:return dict(aborted_at_m='tracking lost before replay', worst_deviation_cm=None, robot_s=0., trace=[])
        T_map_exec = flat(P)@np.linalg.inv(encoders(self.m, self.d)); T_exec_rec = np.linalg.inv(T_map_exec)@T_map_rec
        rep = rebase_planar_commands(ctrl, names, T_exec_rec)
        th = float(self.d.joint('robot_0/base_theta').qpos[0]); rep[:, bi[2]] += 2*math.pi*round((th-rep[0, bi[2]])/(2*math.pi))
        ids = base_cmd_ids(self.m); worst = 0.; aborted = None; t0 = float(self.d.time); trace = []; lost = 0
        # ease in: playback rate ramps 0 -> 1 over EASE_S (joining mid-route from a standstill otherwise
        # lurches to cruise speed and loses SLAM tracking); after that, the recorded commands at their own rate
        dt = self.m.opt.timestep; n_ease = int(EASE_S/dt)
        path_map = rep[:, bi[:2]]@T_map_exec[:2, :2].T+T_map_exec[:2, 3]
        # lead-in: slide (holonomic base targets) from the current encoders onto the route's first command
        cur = np.array([self.d.joint(f'robot_0/{n}').qpos[0] for n in ('base_x', 'base_y', 'base_theta')])
        gap = float(np.linalg.norm(rep[0, bi[:2]]-cur[:2])); n_lead = int(max(.5, gap/.25)/dt)
        for k in range(n_lead):
            w = (k+1)/n_lead; self.d.ctrl[ids] = cur+(rep[0, bi]-cur)*w; mujoco.mj_step(self.m, self.d)
            if (k+1) % 50 == 0:self.R.sense()
        idx = playback_index(rep[:, bi], dt, n_ease)
        for step, f in enumerate(idx):
            i = int(f); w = f-i; u = rep[i, bi]*(1-w)+rep[i+1, bi]*w; k = i
            self.d.ctrl[ids] = u; mujoco.mj_step(self.m, self.d)
            if (step+1) % 50 == 0:
                self.R.sense(); P = self.nav.pose()
                if P is None:
                    lost += 1
                    if lost > 10:aborted = 'tracking lost'; break
                    continue
                lost = 0
                ref = (T_map_exec@planar(*rep[k, bi]))[:2, 3]
                # off the path, not behind on it: distance to the nearest commanded point within +-2 s
                w0, w1 = max(0, k-1000), min(len(path_map), k+1000)
                dev = float(np.min(np.linalg.norm(path_map[w0:w1:10]-P[:2, 3], axis=1))); worst = max(worst, dev)
                enc = np.array([self.d.joint(f'robot_0/{n}').qpos[0] for n in ('base_x', 'base_y', 'base_theta')])
                if len(trace) < 40:trace.append(dict(t=round(k*self.m.opt.timestep, 2), slam=P[:2, 3].round(3).tolist(), ref=ref.round(3).tolist(),
                                                     cmd=rep[k, bi].round(3).tolist(), enc=enc.round(3).tolist(), dev=round(dev, 3)))
                if dev > ABORT_M and step > n_ease:aborted = round(dev, 3); break
        self.R.base = OnboardBaseActuator(self.m, self.d, motion_guard=self.R.guard)
        for _ in range(25):self.R.tick(0., 0.)
        return dict(worst_deviation_cm=round(worst*100, 1), aborted_at_m=aborted, robot_s=round(float(self.d.time)-t0, 1), trace=trace)

    def go(self, location, skills, exclude=None, _replanned=False, prefer_replay=False, standoff_m=0., edge_first=False, prefer=None, kind=None):
        """To within NEAR of a confident remembered manipulation start at `location`. Returns a timed log."""
        t_start = float(self.d.time); log = dict(location=location)
        starts = self.manip_starts(location, skills, exclude, kind=kind)
        if kind and not any(kind in str((r.get('arguments') or {}).get('target', '')).lower() for r in starts):
            edge_first = False  # no past pick of this kind here: the row ends belong to other objects -> nearest start
        if standoff_m:
            # stop short of the remembered pose (along its heading): room for the pipeline's own approach and docking
            starts = [dict(r, start=[r['start'][0]-standoff_m*math.cos(r['start'][2]), r['start'][1]-standoff_m*math.sin(r['start'][2]), r['start'][2]])
                      for r in starts]
        log['manip_starts'] = len(starts)
        if not starts:log['result'] = 'no confident manipulation memory at this location'; return log
        P = self.pose()
        if P is None:log['result'] = 'tracking lost'; return log
        here = P[:2, 3]
        tgt = min(starts, key=lambda r: np.linalg.norm(np.asarray(r['start'][:2])-here))
        if prefer is not None and len(starts) > 1:
            # e.g. a place: the remembered place whose past robot held its object most like we hold ours
            scored = [(float(prefer(r)), i) for i, r in enumerate(starts)]
            best = min(scored); tgt = starts[best[1]]; log['chosen_by'] = dict(score=round(best[0], 3), of=len(starts))
        if edge_first and len(starts) >= 3:
            # open surface: stop in front of an END of the row of remembered pick spots (the end nearer the robot), so the
            # pipeline's search finds an edge object first (user rule: edge objects first; clear-and-store 024: stopping
            # mid-row sent the pick to the middle bowl between two neighbours and every grasp collided)
            xy = np.array([r['start'][:2] for r in starts], float); c = xy.mean(0)
            axis = np.linalg.svd(xy-c, full_matrices=False)[2][0]; pr = (xy-c)@axis
            ends = [starts[int(np.argmin(pr))], starts[int(np.argmax(pr))]]
            tgt = min(ends, key=lambda r: np.linalg.norm(np.asarray(r['start'][:2])-here)); log['edge_first'] = True
        goal = np.asarray(tgt['start'][:2])
        tp = time.monotonic(); opts = self.plan(location, goal, exclude); log['plan_wall_s'] = round(time.monotonic()-tp, 2)
        log['options'] = [{k: v for k, v in o.items() if k != 'row'} for o in opts[:4]]
        if not opts:log['result'] = 'no route (A* failed)'; return log
        log['attempts'] = []; self.nav.trace = [] if DEBUG else None
        # try the quickest options; after a failure never retry the same memory, and keep A* as the last resort
        tries, seen = [], set()
        for o in opts:
            key = o.get('memory') or 'astar'
            if key in seen:continue
            seen.add(key); tries.append(o)
        tries = [o for o in tries if o['kind'] != 'astar'][:2]+[o for o in tries if o['kind'] == 'astar'][:1] if len(tries) > 3 else tries
        tries.sort(key=lambda o: o['seconds'])
        lp = getattr(self.nav, 'leaf_pts', None)
        if lp is not None and len(lp):
            # an opened door/drawer stands where the past robot drove: its route (and the slide onto it) must keep the
            # stowed sweep clear of the leaf, else A* (which plans around the leaf) goes first
            def clear(o):
                if o['kind'] != 'replay':return True
                q = np.asarray(o['row']['route'])[o['j']:, :2]; seg = np.vstack([here, q[:40]])
                fine = np.concatenate([np.linspace(seg[i], seg[i+1], 6) for i in range(len(seg)-1)]) if len(seg) > 1 else seg
                return float(np.min(np.linalg.norm(fine[:, None]-lp[None], axis=2))) >= LEAF_SWEEP_M
            blocked = [o for o in tries if not clear(o)]
            if blocked:log['replays_blocked_by_open_leaf'] = len(blocked)
            tries = [o for o in tries if o not in blocked]
            prefer_replay = prefer_replay and not blocked
        # the robot already stands near the goal: a remembered route (built from elsewhere) would only swing the base
        # around in the tight spot (meal run 2026-11-23: a route replay right after the microwave place turned ~100 deg
        # beside the counter, hit the lower cabinet door and the physics blew up) -> the short direct move only
        if float(np.linalg.norm(goal-here)) < NEAR_GOAL_DIRECT_M:
            direct = [o for o in tries if o['kind'] == 'astar']
            if direct:log['direct_near_goal_m'] = round(float(np.linalg.norm(goal-here)), 2); tries = direct; prefer_replay = False
        # remembered routes must not turn inside the docking band (observed floor next to furniture, not strict free floor)
        def turns_in_band(o):
            if o['kind'] != 'replay':return False
            q = np.asarray(o['row']['route'])[o['j']:]; th0 = float(q[0, 2])
            for x in q[:200:4]:
                c = self.nav.cell(x[:2])
                inside = 0 <= c[0] < self.nav.free.shape[0] and 0 <= c[1] < self.nav.free.shape[1]
                if inside and not self.nav.free[c] and abs(math.atan2(math.sin(x[2]-th0), math.cos(x[2]-th0))) > .3:return True
            return False
        turning = [o for o in tries if turns_in_band(o)]
        if turning and len(turning) < len(tries):log['replays_turning_near_furniture'] = len(turning); tries = [o for o in tries if o not in turning]
        if prefer_replay:
            # right after a fixture operation the robot stands in a tight spot: leave it the way a past robot did
            near = [o for o in tries if o['kind'] == 'replay' and o.get('join_m', 9) <= SLIDE_M]
            tries = near+[o for o in tries if o not in near]
        for best in tries:
            t = float(self.d.time); att = dict(kind=best['kind'], memory=best.get('memory'), estimated_s=round(best['seconds'], 1))
            if best['kind'] == 'replay':
                q = np.asarray(best['row']['route']); j = best['j']
                if best['join_m'] > SLIDE_M:
                    att['join'] = self.nav.drive_to(q[j, :2], tol=.12, deadline_s=max(20., 3*best['join_m']/JOIN_SPEED))
                if att.get('join', 'arrived') == 'arrived':
                    if best['join_m'] > SLIDE_M:self.nav.turn_to(float(q[j, 2]), deadline_s=6.)
                    rp = self.replay_from(best['row'], j); att['replay'] = {k: v for k, v in rp.items() if k != 'trace'}
                    ok = not rp['aborted_at_m']
                else:ok = False
            else:
                att['astar'] = self.nav.drive_to(goal, tol=NEAR*.6, deadline_s=max(20., 3*best['seconds'])); ok = att['astar'] == 'arrived'
            att['robot_s'] = round(float(self.d.time)-t, 1); att['ok'] = bool(ok); log['attempts'].append(att)
            if DEBUG:att['drive_trace'] = self.nav.trace[:]; self.nav.trace = []
            P = self.pose()
            if P is not None and min(float(np.linalg.norm(np.asarray(r['start'][:2])-P[:2, 3])) for r in starts) <= NEAR:break
            if ok:break
        log['chosen'] = log['attempts'][0]['kind'] if log['attempts'] else None
        if log['attempts'] and not log['attempts'][-1]['ok'] and not _replanned and self.pose(wait_s=5.) is not None and any(
                'tracking' in str(x.get('astar') or x.get('join') or (x.get('replay') or {}).get('aborted_at_m')) for x in log['attempts']):
            # tracking dropped and came back: plan again once from where the robot is now
            again = self.go(location, skills, exclude, _replanned=True, prefer_replay=prefer_replay, standoff_m=standoff_m, edge_first=edge_first, kind=kind); again['replanned_after_tracking_loss'] = log['attempts']; return again
        # nearest confident manipulation start from where we are now; coarse A* only if farther than NEAR
        P = self.pose()
        if P is None:log['result'] = 'tracking lost'; return log
        here = P[:2, 3]
        tgt = min(starts, key=lambda r: np.linalg.norm(np.asarray(r['start'][:2])-here)); dist = float(np.linalg.norm(np.asarray(tgt['start'][:2])-here))
        if dist > NEAR:
            t = float(self.d.time); log['astar'] = self.nav.drive_to(np.asarray(tgt['start'][:2]), tol=NEAR*.6, deadline_s=max(20., 3*dist/ASTAR_SPEED))
            log['astar_robot_s'] = round(float(self.d.time)-t, 1)
            P = self.pose()
            if P is None:log['result'] = 'tracking lost'; return log
            here = P[:2, 3]
            tgt = min(starts, key=lambda r: np.linalg.norm(np.asarray(r['start'][:2])-here)); dist = float(np.linalg.norm(np.asarray(tgt['start'][:2])-here))
        if dist <= NEAR and FACE_REMEMBERED:
            # face the way the past robot faced at this start (grocery 2026-11-23: arrived 28 deg off and the baseline's
            # receptacle look did not see the orange basket; a remembered start's heading shows what it saw)
            yaw0 = math.atan2(P[1, 0], P[0, 0]); err0 = math.atan2(math.sin(tgt['start'][2]-yaw0), math.cos(tgt['start'][2]-yaw0))
            if abs(math.degrees(err0)) > 8.:
                log['faced_remembered_deg'] = round(math.degrees(err0), 1); self.nav.turn_to(float(tgt['start'][2]), deadline_s=6.)
                P2 = self.pose()
                if P2 is not None:P = P2
        yaw_err = abs(math.degrees(math.atan2(math.sin(tgt['start'][2]-math.atan2(P[1, 0], P[0, 0])), math.cos(tgt['start'][2]-math.atan2(P[1, 0], P[0, 0])))))
        log.update(manip_memory=f"{tgt['episode']}/{tgt['segment']}", to_manip_start_cm=round(dist*100, 1), yaw_deg=round(yaw_err, 1),
                   ready=bool(dist <= NEAR), robot_s=round(float(self.d.time)-t_start, 1), target_row=tgt)
        log['result'] = 'ready for manipulation replay' if dist <= NEAR else 'not within 30 cm'
        return log
