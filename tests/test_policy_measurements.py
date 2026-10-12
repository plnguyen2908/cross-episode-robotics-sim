"""Policy scoring must handle held objects and contacts on asset child bodies."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cross_episode_sim.env.tasks import breakfast_after_step, finger_contact_bodies, room_of


@pytest.mark.parametrize('reverse', [False, True])
def test_nested_object_and_pad_contacts_count_as_holding_object(reverse):
    # world -> pad -> pad mesh; world -> cup -> handle mesh
    ids = {'robot/pad': 1, 'cup': 3}
    model = SimpleNamespace(nbody=5, body_parentid=[0, 0, 1, 0, 3], geom_bodyid=[2, 4],
                            body=lambda name: SimpleNamespace(id=ids[name]))
    data = SimpleNamespace(ncon=1, contact=[SimpleNamespace(geom1=int(reverse), geom2=int(not reverse))])
    c = SimpleNamespace(model=model, data=data, profile=SimpleNamespace(namespace='robot/', finger_bodies=['pad']),
                        manifest=dict(bindings=[dict(body='cup')]), phase='GATHER', filled=lambda: False,
                        current_room=lambda info: 'kitchen', is_filling_site=lambda room: room == 'kitchen',
                        fill_physical_contents=Mock(), configure_phase=Mock())
    assert finger_contact_bodies(c) == {3}
    breakfast_after_step(c)
    c.fill_physical_contents.assert_not_called()
    data.ncon = 0  # released; cached ancestry must still use fresh contacts
    breakfast_after_step(c)
    c.fill_physical_contents.assert_called_once()
    c.configure_phase.assert_called_once_with('SERVE')


def test_unsupported_vessel_has_no_room_and_does_not_trigger_filling():
    c = SimpleNamespace(current_room=Mock(side_effect=RuntimeError('Missing or ambiguous native table support')))
    assert room_of(c, {'body': 'cup'}) is None


def test_role_verifier_checks_only_that_object_and_handles_in_flight_state():
    from cross_episode_sim.tasks.breakfast.episode import BreakfastEpisode
    c = SimpleNamespace(manifest=dict(bindings=[dict(role='cup', body='cup_body', initial_quaternion=[1, 0, 0, 0],
                                                    destination='dining', destination_position=[0, 0, 1])]),
                        assignment=Mock(return_value={'cup_body': 'table'}), report={},
                        data=SimpleNamespace(body=lambda name: SimpleNamespace(xmat=np.eye(3).ravel(), xpos=np.array([0, 0, 1]))),
                        task_supports={'dining': 'table'}, attached=False, holding_loaf=False)
    assert BreakfastEpisode.verify_role(c, 'cup')
    c.assignment.assert_called_once_with(objects=['cup_body'])
    c.assignment.side_effect = RuntimeError('Missing native table support')
    assert not BreakfastEpisode.verify_role(c, 'cup')
