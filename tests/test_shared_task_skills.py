"""Composite/standalone skill identity, measured goals, and transaction boundaries."""

import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import pytest

from cross_episode_sim.fixtures.portafilter import prepare_attachment
from cross_episode_sim.skills.atomic import BoundSkill
from cross_episode_sim.skills.checkpoint import ExecutionCheckpoint
from cross_episode_sim.skills.coffee import (
    ParkDosingBoxSkill,
    PickDosingBoxSkill,
    PourGroundsSkill,
    coffee_operations,
)
from cross_episode_sim.skills.coffee_cups import PlaceCoffeeCupSkill, RetrieveCoffeeCupSkill
from cross_episode_sim.skills.portafilter import AttachPortafilterSkill, DetachPortafilterSkill, detached
from cross_episode_sim.skills.sequence import SkillSequence
from cross_episode_sim.tasks.coffee.brew import MachineCycle
from cross_episode_sim.tasks.coffee.machine import PORTAFILTER
from cross_episode_sim.tasks.coffee.native_task import CoffeeEpisode
from tests.test_speculative_execution import controller, state


def test_coffee_registry_composes_dosing_and_uses_standalone_portafilter_skills():
    c = SimpleNamespace(wait_machine_ready=Mock(), finish_brew=Mock(), next_mug=Mock())
    ops = coffee_operations(c, Mock())
    assert type(ops["remove"]) is DetachPortafilterSkill
    assert type(ops["reinstall"]) is AttachPortafilterSkill
    assert type(ops["cup"]) is PlaceCoffeeCupSkill
    assert type(ops["retrieve"]) is RetrieveCoffeeCupSkill
    sequence = ops["dose"].factory()
    assert [type(skill) for skill, _ in sequence.steps] == [PickDosingBoxSkill, PourGroundsSkill, ParkDosingBoxSkill]
    assert all(skill.controller is c for skill, _ in sequence.steps)


def test_failed_second_skill_preserves_completed_first_skill(tmp_path):
    c = controller(tmp_path)
    saved = []

    def pick():
        c.step(0.2)
        saved.append(state(c).copy())

    def fail_place():
        c.step(-0.2)
        raise RuntimeError("placement obstructed")

    sequence = SkillSequence(
        c,
        "pick_place",
        [
            (BoundSkill(c, "pick", pick, lambda: True), {}),
            (BoundSkill(c, "place", fail_place, lambda: False), {}),
        ],
    )
    with pytest.raises(RuntimeError, match="placement obstructed"):
        sequence.run()
    np.testing.assert_array_equal(state(c), saved[0])
    assert [e["success"] for e in c.report["skill_execution"]] == [True, False]
    assert c.report["skill_execution"][1]["wall_seconds"] >= 0
    assert c.report["accepted_trials"][0]["phase"] == "pick"


def test_sequence_validates_later_arguments_before_first_motion(tmp_path):
    c = controller(tmp_path)
    action = Mock()
    first = BoundSkill(c, "first", action, lambda: True)
    second = BoundSkill(c, "second", lambda target: None, lambda target: True)
    with pytest.raises(TypeError):
        SkillSequence(c, "invalid", [(first, {}), (second, {"typo": 1})]).run()
    action.assert_not_called()


def test_coffee_checkpoint_restores_machine_report_alias_visuals_and_files(tmp_path):
    base = controller(tmp_path)
    c = CoffeeEpisode.__new__(CoffeeEpisode)
    c.__dict__.update(vars(base))
    c.spec = {"coffee_mugs": ["one", "two"]}
    c.brew = MachineCycle()
    c.machine_cycle = MachineCycle(powered_at=1.0)
    c.report["machine_cycle"] = vars(c.machine_cycle)
    c.workflow_events = []
    path = tmp_path / "workflow_events.json"
    path.write_text("[]")
    checkpoint = ExecutionCheckpoint(c)
    c.machine_cycle.start(4.0, True, True, True)
    c.machine_cycle.update(20.0, True, True, True, True)
    c.brew.power_on(9.0)
    c.workflow_events.append({"event": "brew_complete"})
    path.write_text(json.dumps(c.workflow_events))
    (tmp_path / "cup_placed_2_checkpoint.npz").write_bytes(b"rejected")
    c.model.geom_condim[:] = 4
    checkpoint.restore(c)
    assert c.machine_cycle.powered_at == 1.0
    assert c.machine_cycle.started_at is None and not c.machine_cycle.completed
    assert c.brew.powered_at is None
    assert c.report["machine_cycle"] is vars(c.machine_cycle)
    assert not c.workflow_events
    assert path.read_text() == "[]"
    assert not (tmp_path / "cup_placed_2_checkpoint.npz").exists()
    np.testing.assert_array_equal(c.model.geom_condim, checkpoint.model["geom_condim"])


def portafilter_xml():
    return f'''<mujoco><option timestep=".002"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="counter"><geom type="box" size="1 1 .02" pos="0 0 -.02"/></body>
      <body name="machine" pos="1 0 0"><geom type="box" size=".05 .05 .1"/></body>
      <body name="{PORTAFILTER}" pos="0 0 .03">
        <freejoint name="{PORTAFILTER}_free"/><geom type="box" size=".03 .03 .03" mass=".2"/>
      </body>
    </worldbody><equality>
      <connect name="bayonet_axis_a" body1="{PORTAFILTER}" body2="machine" anchor="0 0 0" active="false"/>
      <connect name="bayonet_axis_b" body1="{PORTAFILTER}" body2="machine" anchor="0 0 .04" active="false"/>
      <weld name="bayonet_locked" body1="{PORTAFILTER}" body2="machine" active="false"/>
    </equality></mujoco>'''


def test_detachment_is_measured_not_a_phase_flag():
    model = mujoco.MjModel.from_xml_string(portafilter_xml())
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data, nstep=1000)
    loading = np.eye(4)
    loading[:3, 3] = data.body(PORTAFILTER).xpos
    c = SimpleNamespace(
        model=model,
        data=data,
        loading=loading,
        counter="counter",
        table_bids={"counter": {model.body("counter").id}},
        lock_eq=model.equality("bayonet_locked").id,
        axis_eq=[model.equality(n).id for n in ("bayonet_axis_a", "bayonet_axis_b")],
        holding_loaf=False,
        attached=False,
        contacts=lambda: [],
        fixture_phase="released",
    )
    assert detached(c)
    data.eq_active[c.lock_eq] = True
    assert not detached(c)
    data.eq_active[:] = False
    c.holding_loaf = True
    assert not detached(c)


def test_attachment_scene_preserves_seated_constraint_frames(tmp_path):
    scene = tmp_path / "robocasa_scene.xml"
    scene.write_text(portafilter_xml())
    original = mujoco.MjModel.from_xml_path(str(scene))
    loading = np.eye(4)
    loading[:3, 3] = [0.3, 0.2, 0.03]
    manifest = {"coffee": {"portafilter_reference_pose": loading.tolist()}, "bindings": [{"body": PORTAFILTER}]}
    prepare_attachment(tmp_path, manifest)
    for name, values in manifest["coffee"]["bayonet_constraint_data"].items():
        np.testing.assert_array_equal(values, original.eq_data[original.equality(name).id])
    tree = ET.parse(scene)
    body = tree.find(f".//body[@name='{PORTAFILTER}']")
    np.testing.assert_allclose(np.fromstring(body.get("pos"), sep=" "), loading[:3, 3] - [0, 0, 0.002])
    assert all(e.get("active") == "false" for e in tree.find("equality"))


@pytest.mark.parametrize("action", ["detach", "attach"])
def test_standalone_portafilter_cli_uses_shared_workflow_registry(action):
    from cross_episode_sim.fixtures.portafilter import main

    with patch("cross_episode_sim.tasks.coffee.workflow.main", return_value=0) as run:
        assert main(["--action", action, "--output", "runs/test"]) == 0
    run.assert_called_once_with(["--atomic-skill", action + "-portafilter", "--output", "runs/test"])


def test_public_spout_pick_tries_different_paths_from_same_checkpoint(tmp_path):
    from cross_episode_sim.skills.breakfast_interface import breakfast_skill_interface
    from cross_episode_sim.skills.coffee_cups import PickCupFromSpoutSkill
    c = controller(tmp_path)
    c.manifest = dict(bindings=[dict(role='coffee_mug', body='cup')], coffee=dict(coffee_mugs=['cup']))
    c.coffee_mugs_roles = lambda: ['coffee_mug']
    c.wait_machine_ready = Mock(); c.finish_brew = Mock()
    c.info = lambda role: c.manifest['bindings'][0]
    c.object_name = 'cup'; c.holding_loaf = False; c.attached = False
    c.contacts = lambda: ['left', 'right'] if c.holding_loaf else []
    starts, candidates = [], []

    def extract(skill, candidate):
        starts.append(state(c).copy()); candidates.append(candidate)
        c.step(.2)
        if candidate == 0:
            raise RuntimeError('Actual-mesh collision in the original extraction path')
        c.holding_loaf = True

    with patch.object(PickCupFromSpoutSkill, 'execute_candidate', extract):
        breakfast_skill_interface(c).bindings['pick'].skill.run(role='coffee_mug', source='spout')
    assert candidates == [0., .004]
    np.testing.assert_array_equal(starts[0], starts[1])
    assert c.holding_loaf and c.report['accepted_trials'][-1]['attempts'] == 2
    assert (tmp_path/'rejected_trials/0001_pick/failure_state.npz').exists()
