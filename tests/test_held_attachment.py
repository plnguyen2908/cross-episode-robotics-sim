import numpy as np

from cross_episode_sim.perception.held_attachment import connected_attachment_surface


def patch(end=.1):
    x, y = np.meshgrid(np.arange(0, end, .005), np.arange(0, .08, .005))
    return np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]


def test_disconnected_background_mask_pixels_do_not_expand_attachment():
    prior = patch()
    observed = np.concatenate([prior+[0., 0., .01], prior+[.5, 0., .2]])
    accepted, evidence = connected_attachment_surface(observed, prior)
    assert len(accepted) == len(prior)
    assert np.allclose(accepted[:, 2], .01)
    assert any(not e['accepted'] for e in evidence)


def test_connected_large_surface_is_rejected_instead_of_trimmed_to_fit():
    accepted, _ = connected_attachment_surface(patch(.5), patch())
    assert len(accepted) == 0


def test_too_sparse_observation_cannot_replace_attachment():
    accepted, evidence = connected_attachment_surface(patch()[:12], patch())
    assert accepted.shape == (0, 3)
    assert evidence == []
