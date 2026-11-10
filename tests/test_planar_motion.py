import numpy as np
from cross_episode_sim.slam.planar_motion import PlanarVisualMotion


def pose(x=0.,y=0.,angle=0.):
    T=np.eye(4);c,s=np.cos(angle),np.sin(angle);T[:2,:2]=[[c,-s],[s,c]];T[:2,3]=[x,y];return T


def test_turn_does_not_acquire_spurious_sideways_translation():
    motion=PlanarVisualMotion();motion.observe(pose())
    motion.command(0.,.6,.1)
    result=motion.observe(pose(.6,.2,.06))
    np.testing.assert_allclose(result[:2,3],0.)
    np.testing.assert_allclose(result[:2,:2],pose(angle=.06)[:2,:2])


def test_command_does_not_count_as_measured_progress_when_blocked():
    motion=PlanarVisualMotion();motion.observe(pose())
    motion.command(.4,0.,.1)
    np.testing.assert_array_equal(motion.observe(pose()),pose())


def test_observed_forward_motion_and_map_correction_are_retained():
    motion=PlanarVisualMotion();motion.observe(pose())
    motion.command(.4,0.,.1)
    np.testing.assert_allclose(motion.observe(pose(.035,.03))[:2,3],[.035,0.])
    motion.correct(pose(.05,.02));motion.command(.4,0.,.1)
    np.testing.assert_allclose(motion.observe(pose(.07,.03))[:2,3],[.085,.02])
