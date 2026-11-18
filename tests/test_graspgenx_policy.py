import numpy as np
from cross_episode_sim.control.graspgenx_policy import (
    PAD_CENTRE_IN_TOOL, endpoint_proposals, split_plausible, graspgenx_to_tcp, proposal_rows, tcp_to_m2t2_root)
from cross_episode_sim.control.m2t2_policy import grasp_to_tcp, valid_grasp_poses


def pose(seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    out = np.eye(4)
    out[:3, :3] = q
    out[:3, 3] = rng.normal(size=3)
    return out


def test_tool_frame_keeps_approach_and_maps_closing_axis():
    grasp = pose(0)
    tcp = graspgenx_to_tcp(grasp)
    assert np.allclose(tcp[:3, 2], grasp[:3, 2])
    assert np.allclose(tcp[:3, 1], grasp[:3, 0])
    assert np.allclose(tcp[:3, 3], grasp[:3, 3] + .145*grasp[:3, 2])


def test_m2t2_root_round_trips_through_existing_conversion():
    tcp = graspgenx_to_tcp(pose(1))
    assert np.allclose(grasp_to_tcp(tcp_to_m2t2_root(tcp)), tcp)


def test_rows_use_the_executor_schema_and_observed_contacts():
    surface = np.random.default_rng(2).normal(scale=.03, size=(200, 3))
    rows = proposal_rows([pose(3), pose(4)], [.9, .4], ['diff', 'obb'], surface, sampling_pass=1)
    valid, rejected = valid_grasp_poses(rows)
    assert len(valid) == 2 and not rejected
    assert rows[0]['group'] == 10000 and rows[1]['proposal_source'] == 'graspgenx_obb'
    for row in rows:
        tcp = np.asarray(row['tcp'])
        centre = tcp[:3, 3] + tcp[:3, :3] @ PAD_CENTRE_IN_TOOL
        nearest = surface[np.linalg.norm(surface-centre, axis=1).argmin()]
        assert np.allclose(row['contact'], nearest)


class ScoredEndpoint:
    """Stands in for the graspgenx service: fixed scores per sampling pass."""
    def __init__(self, scores):
        self.scores = scores
        self.requests = []

    def call(self, object_points, seed):
        self.requests.append((len(object_points), seed))
        scores = self.scores[seed]
        return dict(grasps=[pose(10*seed+i) for i in range(len(scores))], confidence=scores, tags=['diff']*len(scores))


def test_endpoint_proposals_keep_confident_grasps_or_the_best_few():
    surface = np.random.default_rng(5).normal(scale=.03, size=(200, 3))
    endpoint = ScoredEndpoint([[.9, .2, .8], [.75, .1]])
    rows = endpoint_proposals(endpoint, surface, minimum=3)
    assert endpoint.requests == [(200, 0), (200, 1)]
    assert [row['confidence'] for row in rows] == [.9, .8, .75]
    assert [row['sampling_pass'] for row in rows] == [0, 0, 1]
    # Too few confident grasps: the best ones are kept in order, never an empty list.
    weak = endpoint_proposals(ScoredEndpoint([[.3, .6], [.1]]), surface, minimum=2)
    assert [row['confidence'] for row in weak] == [.6, .3]


def test_fallback_tier_leaves_the_confident_rows_unchanged():
    surface = np.random.default_rng(5).normal(scale=.03, size=(200, 3))
    scores = [[.9, .2, .8], [.75, .4], [.95, .35, .1]]
    plain = endpoint_proposals(ScoredEndpoint(scores), surface, minimum=3)
    endpoint = ScoredEndpoint(scores)
    rows = endpoint_proposals(endpoint, surface, minimum=3, fallback_passes=3)
    assert [seed for _, seed in endpoint.requests] == [0, 1, 2]
    confident = [row for row in rows if not row['fallback']]
    assert [{**row, 'fallback': None} for row in confident] == [{**row, 'fallback': None} for row in plain]
    # Lower-scored samples and every sample of the extra seeds follow, best first, above the floor.
    assert [(row['sampling_pass'], row['confidence']) for row in rows[len(confident):]] == [(2, .95), (1, .4), (2, .35)]


def test_split_plausible_screens_blocked_and_far_side_wrist_poses():
    def row(name, position, approach):
        tcp = np.eye(4)
        z = np.asarray(approach, float)/np.linalg.norm(approach)
        x = np.cross([0., 1., 0.], z) if abs(z[1]) < .9 else np.cross([1., 0., 0.], z)
        x /= np.linalg.norm(x)
        tcp[:3, :3] = np.c_[x, np.cross(z, x), z]
        tcp[:3, 3] = position
        return dict(name=name, tcp=tcp.tolist())
    # Gripper body reaches 8 cm behind the tool point and 3 cm to each side of it.
    envelope = np.array([[sx*.03, sy*.03, z] for sx in (-1, 1) for sy in (-1, 1) for z in (-.08, 0.)])
    counter = dict(centers=[[.6, 0., .45]], half=[[.3, .5, .45]], rotations=[np.eye(3)])
    rows = [row('low_side', [.6, 0., .93], [1, 0, 0]), row('from_behind', [.66, 0., 1.0], [-1, 0, 0]),
            row('high_side', [.6, 0., .99], [1, 0, 0]), row('top_down', [.6, 0., 1.02], [0, 0, -1])]
    plausible, blocked = split_plausible(rows, envelope, shoulder=[0., 0., .9], **counter)
    assert [r['name'] for r in plausible] == ['high_side', 'top_down']
    assert [r['name'] for r in blocked] == ['low_side', 'from_behind']


def scene():
    rng = np.random.default_rng(7)
    cup = rng.uniform([.56, -.04, .93], [.64, .04, 1.05], (3000, 3))
    counter = np.c_[rng.uniform(.35, .85, 5000), rng.uniform(-.2, .2, 5000), np.full(5000, .915)]
    points = np.vstack([cup, counter])
    return dict(points=points, rgb=np.zeros_like(points), target=np.array([.60, 0., .99]))


def run_script(monkeypatch, tmp_path, script, endpoint, extra=()):
    import json, runpy, sys
    np.savez(tmp_path/'input.npz', **scene())
    monkeypatch.setattr('cross_episode_sim.perception.rpc.configured_client',
                        lambda kind: endpoint if kind == 'graspgenx' else None)
    # The M2T2 endpoint entry only keeps the worker from reserving GPU memory in this test.
    monkeypatch.setenv('SENSOR_MODEL_ENDPOINTS', json.dumps(dict(m2t2='unused', graspgenx='fake')))
    monkeypatch.setattr('cross_episode_sim.control.m2t2_policy.M2T2Policy',
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('M2T2 loaded for a pick')))
    monkeypatch.setattr(sys, 'argv', [script, '--input', str(tmp_path/'input.npz'),
                                      '--output', str(tmp_path/'predictions.json'), *extra])
    try:
        runpy.run_path(script, run_name='__main__')
    except SystemExit as stop:
        assert not stop.code
    return json.loads((tmp_path/'predictions.json').read_text())


def test_shared_pick_worker_uses_the_configured_graspgenx_endpoint(monkeypatch, tmp_path):
    # Cup picks and confined picks call this worker; no M2T2 model may be built for them.
    endpoint = ScoredEndpoint([[.9, .8], [.75], [.6], [.2], [.5], []])
    report = run_script(monkeypatch, tmp_path, 'scripts/m2t2_worker.py', endpoint, ['--checkpoint', 'unused'])
    assert report['proposal_model'] == 'graspgenx'
    # The confident tier is what two seeds gave before; four more seeds only add marked fallback rows.
    assert [(row['confidence'], row['fallback']) for row in report['candidates']] == [
        (.9, False), (.8, False), (.75, False), (.6, True), (.5, True)]
    assert report['fallback_candidates'] == 2
    # Only the segmented cup is sent, never the counter.
    assert endpoint.requests and all(0 < count <= 3000 for count, _ in endpoint.requests)
    assert valid_grasp_poses(report['candidates'])[0]


def test_cookware_proposals_use_the_configured_graspgenx_endpoint(monkeypatch, tmp_path):
    endpoint = ScoredEndpoint([[.9, .8], [.75]])
    report = run_script(monkeypatch, tmp_path, 'scripts/cookware_grasp_proposals.py', endpoint)
    assert report['proposal_model'] == 'graspgenx' and endpoint.requests
    assert {row['proposal_source'].split('_')[0] for row in report['candidates']} >= {'graspgenx'}


class FacingEndpoint:
    """Returns one grasp approaching along -x and one along +x per pass."""
    def __init__(self):
        self.requests = []

    def call(self, object_points, seed):
        self.requests.append((len(object_points), seed))
        grasps = []
        for sign in (-1., 1.):
            grasp = np.eye(4)
            grasp[:3, :3] = np.column_stack([[0., 0., 1.], [0., sign, 0.], [sign, 0., 0.]])
            grasp[:3, 3] = [-sign*.145, 0., .01*seed]
            grasps.append(grasp)
        return dict(grasps=grasps, confidence=[.4+.1*seed, .9], tags=['diff', 'diff'])


def test_handle_proposals_send_the_handle_only_and_keep_grasps_facing_the_panel():
    from cross_episode_sim.control.graspgenx_policy import handle_proposals
    handle = np.column_stack([np.zeros(40), np.zeros(40), np.linspace(-.1, .1, 40)])
    endpoint = FacingEndpoint()
    rows = handle_proposals(endpoint, handle, [1., 0., 0.])
    # A sparse handle is repeated up to the service minimum, never extended.
    assert endpoint.requests == [(120, 0), (120, 1), (120, 2)]
    # The grasp approaching through the door (along the outward normal) is dropped
    # despite its higher score; the rest come most confident first.
    assert [round(row['confidence'], 1) for row in rows] == [.6, .5, .4]
    for row in rows:
        assert np.asarray(row['tcp'])[:3, 2] @ np.array([1., 0., 0.]) < -.99
        assert np.linalg.norm(handle - np.asarray(row['contact']), axis=1).min() < 1e-9
    assert handle_proposals(endpoint, handle[:5], [1., 0., 0.]) == []


def test_grasps_slide_along_a_handle_bar_to_the_pointed_spot():
    from cross_episode_sim.control.graspgenx_policy import slide_along_handle
    bar = np.column_stack([np.zeros(60), np.zeros(60), np.linspace(0., .2, 60)])
    tcp = np.eye(4)
    tcp[:3, :3] = np.column_stack([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    tcp[:3, 3] = [.015, 0., .15]
    rows = [dict(tcp=tcp.tolist(), contact=[0., 0., .15], confidence=.9)]
    moved = slide_along_handle(rows, bar, [0., 0., .02])[0]
    out = np.asarray(moved['tcp'])
    assert np.allclose(out[:3, :3], tcp[:3, :3])
    assert np.isclose(out[2, 3], .02) and np.isclose(moved['handle_slide'], -.13)
    assert abs(moved['contact'][2] - .02) < .004
    # A spot beyond the bar end is clipped to the observed end.
    assert np.isclose(np.asarray(slide_along_handle(rows, bar, [0., 0., .5])[0]['tcp'])[2, 3], .2)
    # A compact (non-bar) handle is left unchanged.
    knob = np.random.default_rng(0).normal(size=(60, 3))*.01
    assert slide_along_handle(rows, knob, [0., 0., .02]) == rows


def test_handle_grasps_are_offered_at_the_pointed_spot_and_the_bar_middle():
    from cross_episode_sim.control.graspgenx_policy import handle_proposals
    bar = np.column_stack([np.zeros(120), np.zeros(120), np.linspace(0., .24, 120)])
    endpoint = FacingEndpoint()
    rows = handle_proposals(endpoint, bar, [1., 0., 0.], passes=1, target=[0., 0., .02])
    heights = {row['handle_slide_target']: round(float(np.asarray(row['tcp'])[2, 3]), 2) for row in rows}
    assert heights == {'pointed': .02, 'middle': .12}
    assert len(rows) == 2


def test_handle_proposals_fall_back_when_the_service_finds_the_handle_too_sparse():
    import numpy as np
    from cross_episode_sim.control.graspgenx_policy import handle_proposals
    class Client:
        def call(self,**kw):raise RuntimeError('ValueError: sample requires at least 100 object points; received 79')
    surface=np.random.default_rng(0).normal(0,.01,(150,3)).astype(np.float32)
    assert handle_proposals(Client(),surface,np.array([1.,0.,0.]))==[]
