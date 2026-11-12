import numpy as np
import pytest
from cross_episode_sim.code_policy.closing_redock import closing_face
from cross_episode_sim.runtime.fixture_controls import handle_description


@pytest.mark.parametrize('sign',[-1,1])
def test_closing_dock_tracks_opposite_hinge_sides_in_map_frame(sign):
    hinge=np.array([2.,-3.,1.])
    memory=dict(hinge=hinge,closed_handle=hinge+[0.,.6,0.],
                open_handle=hinge+[-sign*.6,0.,0.],closed_normal=np.array([-1.,0.,0.]))
    point,normal=closing_face(memory)
    np.testing.assert_allclose(point,memory['open_handle'])
    np.testing.assert_allclose(normal,[0.,-sign,0.],atol=1e-12)
    # Goal must be outside the open face, not at the closed appliance dock.
    goal=point+.65*normal
    assert (goal-point)@normal==pytest.approx(.65)


def test_close_prompt_does_not_identify_closed_edge_beside_keypad():
    assert 'immediately left of the keypad' in handle_description('microwave',True)
    assert 'immediately left of the keypad' not in handle_description('microwave',False)
    assert 'OPEN' in handle_description('microwave',False)
    assert 'OPEN' in handle_description('fridge',False)
