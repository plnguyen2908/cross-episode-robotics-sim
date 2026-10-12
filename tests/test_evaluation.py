"""Evaluator isolation, backend routing, measured scoring, and budget behavior."""
import json
from pathlib import Path

import numpy as np
import pytest

from cross_episode_sim.evaluation import Budgets, evaluate


class Session:
    supports_actions = True

    def __init__(self):
        self.time = 0.; self.solved = False; self.oracle_calls = []; self.actions = []

    def observe(self):
        return dict(state=np.zeros(11), images={'wrist': np.zeros((2, 2, 3), np.uint8)},
                    sim_time=self.time, goals={'hidden_goal': self.solved}, controller='secret')

    def catalog(self):
        return [dict(name='pick', parameters=dict(type='object', properties={'role': {'enum': ['cup']}}))]

    def validate_skill(self, name, arguments):
        if name != 'pick' or arguments != {'role': 'cup'}:
            raise ValueError('invalid skill')

    def oracle_skill(self, name, arguments):
        self.oracle_calls.append((name, arguments)); self.time += 1; self.solved = True
        return dict(success=True, preconditions={'secret_object_xyz': [1, 2, 3]})

    def step(self, action):
        self.actions.append(action.copy()); self.time += .04
        self.solved = bool(action[0] == 1)
        return dict(terminated=self.solved, truncated=False, failure=None)

    def score(self):
        return dict(physical_goal=self.solved)

    def sim_time(self):
        return self.time

    def action_spec(self):
        return dict(shape=[11])


class Planner:
    def reset(self, context):
        self.context = context

    def plan(self, observation, feedback):
        assert set(observation) == {'images'}
        return dict(skill='pick', arguments={'role': 'cup'})


class VLA:
    def start_skill(self, name, arguments, context):
        self.request = (name, arguments)

    def act(self, observation):
        assert 'state' in observation and 'sim_time' in observation
        return {'actions': [[1, *([0]*10)], [0]*11]}


def test_vlm_oracle_uses_public_interface_and_private_goal(tmp_path):
    s = Session(); model = Planner()
    result = evaluate(model, s, tmp_path/'run', context={'instruction': 'Do the task', 'histories': []})
    assert result['success'] and result['termination'] == 'task_success'
    assert s.oracle_calls == [('pick', {'role': 'cup'})] and not s.actions
    assert 'physical_goal' not in model.context
    assert (tmp_path/'run/observations/000000.npz').exists()


def test_vla_executes_actions_and_never_oracle_motion(tmp_path):
    s = Session(); vla = VLA()
    result = evaluate(Planner(), s, tmp_path/'run', backend='vla', vla=vla)
    assert result['success'] and result['actions'] == 1
    assert len(s.actions) == 1 and not s.oracle_calls  # stops mid-chunk at termination
    assert vla.request == ('pick', {'role': 'cup'})


@pytest.mark.parametrize('backend', ['oracle', 'vla'])
def test_agent_can_use_either_skill_backend(tmp_path, backend):
    class Agent:
        def run(self, context, tools):
            obs = tools.observe()
            assert set(obs) == {'images'}
            answer = tools.execute_skill('pick', {'role': 'cup'})
            assert 'preconditions' not in answer and 'secret_object_xyz' not in str(answer)
            assert answer['terminal']
    s = Session()
    result = evaluate(Agent(), s, tmp_path/'run', planner='agent', backend=backend, vla=VLA() if backend == 'vla' else None)
    assert result['success'] and result['planner'] == 'agent'
    assert bool(s.oracle_calls) == (backend == 'oracle')


def test_agent_completion_claim_is_not_success(tmp_path):
    class Agent:
        def run(self, context, tools):
            tools.finish()
    result = evaluate(Agent(), Session(), tmp_path/'run', planner='agent')
    assert not result['success'] and result['termination'] == 'model_finished'


def test_vla_done_is_unverified_and_does_not_set_task_goal(tmp_path):
    class DoneVLA(VLA):
        def act(self, observation):
            return {'done': True}
    class FinishPlanner(Planner):
        def plan(self, observation, feedback):
            if feedback:
                assert feedback['status'] == 'model_reported_done' and feedback['verification'] == 'unverified'
                return {'finish': True}
            return super().plan(observation, feedback)
    result = evaluate(FinishPlanner(), Session(), tmp_path/'run', backend='vla', vla=DoneVLA())
    assert not result['success'] and result['actions'] == 0


def test_invalid_whole_chunk_is_rejected_before_any_action(tmp_path):
    class Agent:
        def run(self, context, tools):
            assert tools.act([[1, *([0]*10)], [float('nan'), *([0]*10)]])['error'] == 'invalid_action'
            tools.finish()
    s = Session()
    result = evaluate(Agent(), s, tmp_path/'run', planner='agent', backend='direct')
    assert not s.actions and not result['success']


def test_action_budget_counts_vla_actions_not_only_skill_calls(tmp_path):
    class HoldVLA(VLA):
        def act(self, observation):
            return {'actions': [[0]*11]*3}
    s = Session()
    result = evaluate(Planner(), s, tmp_path/'run', backend='vla', vla=HoldVLA(), budgets=Budgets(actions=2))
    assert len(s.actions) == 2 and result['termination'] == 'action_budget' and not result['success']


def test_invalid_calls_are_metered(tmp_path):
    class Invalid(Planner):
        def plan(self, observation, feedback):
            return dict(skill='not_a_skill', arguments={})
    result = evaluate(Invalid(), Session(), tmp_path/'run', budgets=Budgets(tool_calls=4))
    assert result['termination'] == 'tool_budget' and not result['success']
    assert result['tool_calls'] == 4


def test_model_exception_produces_result_and_does_not_execute(tmp_path):
    class Broken(Planner):
        def plan(self, observation, feedback):
            raise RuntimeError('provider unavailable')
    result = evaluate(Broken(), Session(), tmp_path/'run')
    assert not result['success'] and 'provider unavailable' in result['error']
    assert json.loads((tmp_path/'run/result.json').read_text()) == result


def test_unsupported_low_level_task_fails_before_model_calls(tmp_path):
    s = Session(); s.supports_actions = False
    with pytest.raises(ValueError, match='independently evaluated'):
        evaluate(Planner(), s, tmp_path/'run', backend='vla', vla=VLA())


@pytest.mark.parametrize('options', [{'actions': 0}, {'tool_calls': True}, {'wall_seconds': float('nan')}])
def test_invalid_budgets(options):
    with pytest.raises(ValueError):
        Budgets(**options)


def test_oracle_exceeding_simulation_budget_cannot_pass(tmp_path):
    result = evaluate(Planner(), Session(), tmp_path/'run', budgets=Budgets(sim_seconds=.5))
    assert result['goals']['physical_goal']
    assert not result['success'] and result['termination'] == 'simulation_budget'


def test_model_inference_wall_time_is_checked_before_motion(tmp_path, monkeypatch):
    from cross_episode_sim.evaluation import runner
    clock = [0.]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])

    class Slow(Planner):
        def plan(self, observation, feedback):
            clock[0] = 20.
            return super().plan(observation, feedback)
    s = Session()
    result = evaluate(Slow(), s, tmp_path/'run', budgets=Budgets(wall_seconds=10.))
    assert not s.oracle_calls and result['termination'] == 'wall_budget'


def test_agent_return_after_timeout_is_not_success(tmp_path, monkeypatch):
    from cross_episode_sim.evaluation import runner
    clock = [0.]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    s = Session()

    class SlowAgent:
        def run(self, context, tools):
            clock[0] = 20.
            s.solved = True
    result = evaluate(SlowAgent(), s, tmp_path/'run', planner='agent', budgets=Budgets(wall_seconds=10.))
    assert not result['success'] and result['termination'] == 'wall_budget'


def test_success_on_last_allowed_action_is_valid(tmp_path):
    result = evaluate(Planner(), Session(), tmp_path/'run', backend='vla', vla=VLA(), budgets=Budgets(actions=1))
    assert result['success'] and result['actions'] == 1


def test_step_error_is_not_mislabeled_as_invalid_model_action(tmp_path):
    class BrokenSession(Session):
        def step(self, action):
            raise ValueError('broken simulator state')

    class Agent:
        def run(self, context, tools):
            tools.act([0]*11)
    result = evaluate(Agent(), BrokenSession(), tmp_path/'run', planner='agent', backend='direct')
    assert not result['success'] and 'broken simulator state' in result['error']
    assert 'invalid_action' not in (tmp_path/'run/events.jsonl').read_text() if (tmp_path/'run/events.jsonl').exists() else True


def test_public_history_reads_only_given_videos_and_prompt(tmp_path, monkeypatch):
    from cross_episode_sim.evaluation import history
    (tmp_path/'prompt.txt').write_text('Serve breakfast using the histories.')
    (tmp_path/'current.mp4').write_text('SECRET')
    (tmp_path/'run_manifest.json').write_text('SECRET')
    opened, closed = [], []

    class Reader:
        def __init__(self, path):
            self.path = path
            opened.append(path.name)

        def count_frames(self):
            return 5

        def get_meta_data(self):
            return {'fps': 2}

        def get_data(self, index):
            return np.full((2, 2, 3), index, dtype=np.uint8)

        def close(self):
            closed.append(self.path.name)

    monkeypatch.setattr(history.imageio, 'get_reader', Reader)
    for name in ('history_1', 'history_2'):
        (tmp_path/f'{name}.mp4').touch()
    context = history.history_context(tmp_path)
    assert opened == closed == []
    sampled = history.sample_video(Path(context['histories'][0]['video_path']), 3)
    assert opened == closed == ['history_1.mp4']
    assert context['instruction'] == 'Serve breakfast using the histories.'
    assert sampled['timestamps_seconds'] == [0., 1., 2.]
    assert 'SECRET' not in str(context)
    (tmp_path/'history_10.mp4').touch()
    (tmp_path/'history_3.mp4').touch()
    assert [h['name'] for h in history.history_context(tmp_path)['histories']] == ['history_1', 'history_2', 'history_3', 'history_10']


def test_robot_motion_tools_are_metered_and_do_not_call_oracle(tmp_path):
    class Kin:
        limits = np.array([[-2., 2.]]*7)

        def forward(self, joints=None, base=None):
            return dict(tcp_pose_world=np.eye(4).tolist())

        def solve(self, position, quaternion):
            return np.ones(7)*.1

    class RobotSession(Session):
        kinematics = Kin()

        def robot_state(self):
            return np.zeros(11) if not self.actions else self.actions[-1]

        def action_spec(self):
            return dict(shape=[11], control_dt_seconds=.04)

    class Agent:
        def run(self, context, tools):
            assert 'move_ee' in context['robot_tools']
            assert tools.kinematics()['tcp_pose_world'] == np.eye(4).tolist()
            assert tools.move_joints([3]*7)['status'] == 'rejected'
            answer = tools.move_ee([0, 0, 0], [1, 0, 0, 0], duration=.08)
            assert answer['actions'] == 2
            assert tools.gripper(1, duration=.08)['actions'] == 2
            tools.finish()

    s = RobotSession()
    result = evaluate(Agent(), s, tmp_path/'run', planner='agent', backend='direct')
    assert result['error'] is None and result['actions'] == 4 and not s.oracle_calls
    assert np.allclose(s.actions[-1][3:10], .1)
    assert s.actions[-1][10] == 1


def test_cli_records_setup_failures_for_every_episode(tmp_path, monkeypatch):
    from scripts import evaluate_models
    output = tmp_path/'batch'
    monkeypatch.setattr('sys.argv', ['evaluate_models', '--model', 'does_not_exist:Model',
                                   '--episodes', '2', '--output', str(output)])
    assert evaluate_models.main() == 0
    summary = json.loads((output/'summary.json').read_text())
    assert summary['success_rate'] == 0 and len(summary['episodes']) == 2
    for index in range(2):
        result = json.loads((output/f'episode_{index:03d}/result.json').read_text())
        assert result['termination'] == 'evaluation_error' and not result['success']
        assert 'ModuleNotFoundError' in result['error']


def test_concurrent_agent_calls_are_serialized(tmp_path):
    import threading
    import time

    class SerialSession(Session):
        stepping = False

        def step(self, action):
            assert not self.stepping, 'Concurrent mutation of simulator state'
            self.stepping = True
            time.sleep(.005)
            result = super().step(action)
            self.stepping = False
            return result

    class Agent:
        def run(self, context, tools):
            from concurrent.futures import ThreadPoolExecutor
            barrier = threading.Barrier(2)

            def call():
                barrier.wait()
                return tools.act([0]*11)

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(call) for _ in range(2)]
                assert all(f.result()['status'] == 'executed' for f in futures)
            tools.finish()
    result = evaluate(Agent(), SerialSession(), tmp_path/'run', planner='agent', backend='direct')
    assert result['error'] is None and result['actions'] == 2


def test_cli_repeated_output_preserves_previous_run(tmp_path, monkeypatch):
    from scripts import evaluate_models
    output = tmp_path/'batch'; output.mkdir()
    (output/'keep.txt').write_text('previous run')
    monkeypatch.setattr('sys.argv', ['evaluate_models', '--model', 'does_not_exist:Model', '--output', str(output)])
    assert evaluate_models.main() == 0
    assert (output/'keep.txt').read_text() == 'previous run'
    runs = list(tmp_path.glob('batch_*'))
    assert len(runs) == 1 and (runs[0]/'summary.json').exists()
