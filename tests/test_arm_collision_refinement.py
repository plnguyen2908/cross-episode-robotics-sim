import tempfile
from types import SimpleNamespace
import unittest
import mujoco
import numpy as np
from cross_episode_sim.robot.arm_model import export_arm_model


class ConservativeHullCoverTest(unittest.TestCase):
    def test_thin_mast_cover_is_conservative_without_blocking_empty_space(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="robot_0/mast"><geom type="capsule"
          fromto="0 0 .32 0 0 1.7" size=".012"/>
          <site name="robot_0/tcp"/></body></worldbody></mujoco>''')
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        profile = SimpleNamespace(name='test', namespace='robot_0/', arm_joints=(),
                                  torso_joints=(), tcp_site='tcp', travel_posture=())
        with tempfile.TemporaryDirectory() as temp:
            config = export_arm_model(m, d, profile, temp)
        spheres = config['collision_spheres']['mast']
        c = np.array([s['center'] for s in spheres])
        r = np.array([s['radius'] for s in spheres])
        # Entire cylinder surface and hemispherical end caps remain enclosed.
        z = np.linspace(.32, 1.7, 2001)
        cylinder = np.column_stack([np.full_like(z, .012), z * 0, z])
        angle = np.linspace(0, np.pi, 501)
        caps = np.column_stack([.012*np.sin(angle), angle*0,
                               .012*np.cos(angle) + np.where(np.cos(angle)>=0, 1.7, .32)])
        points = np.concatenate([cylinder, caps])
        gaps = np.linalg.norm(points[:, None] - c, axis=-1) - r
        self.assertLessEqual(float(gaps.min(axis=1).max()), 1e-10)
        # A point 30 mm from the pole axis is physically clear and must remain so.
        self.assertGreater(float((np.linalg.norm(c-[.03, 0, 1.], axis=1)-r).min()), .01)
        self.assertLess(float(r.max()), .014)

    def test_refined_cover_preserves_mesh_and_rejects_empty_aabb_corners(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco><asset>
          <mesh name="tetra" vertex="0 0 0 .2 0 0 0 .2 0 0 0 .2"/>
          </asset><worldbody><body name="robot_0/link">
          <joint name="robot_0/joint" type="hinge" range="-90 90"/>
          <geom type="mesh" mesh="tetra"/><site name="robot_0/tcp"/>
          </body></worldbody></mujoco>''')
        d = mujoco.MjData(m); mujoco.mj_forward(m,d)
        profile = SimpleNamespace(name='test', namespace='robot_0/', arm_joints=('joint',),
                                  torso_joints=(), tcp_site='tcp', travel_posture=(0.,))
        with tempfile.TemporaryDirectory() as temp:
            config = export_arm_model(m,d,profile,temp,collision_resolution=.015)
        spheres = config['collision_spheres']['link']
        c=np.array([s['center'] for s in spheres]);r=np.array([s['radius'] for s in spheres])
        rng=np.random.default_rng(0)
        weights=rng.dirichlet(np.ones(4),size=2000)
        vertices=np.array([[0,0,0],[.2,0,0],[0,.2,0],[0,0,.2]])
        points=np.vstack([vertices,weights@vertices])
        gaps=np.linalg.norm(points[:,None,:]-c[None,:,:],axis=-1)-r
        self.assertLessEqual(float(gaps.min(axis=1).max()),1e-6)
        # Far AABB corner is outside the tetrahedron and should stay available.
        self.assertGreater(float((np.linalg.norm(c-[.18,.18,.18],axis=1)-r).min()),.02)

if __name__ == '__main__':
    unittest.main()
