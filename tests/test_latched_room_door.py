import mujoco
import numpy as np
import pytest

from cross_episode_sim.tasks.latched_room_door import install_latched_leaf
from cross_episode_sim.tasks.room_doors import install_room_doors
from tests.test_room_doors import scene


def step(model,data,seconds,hinge,handle):
    data.qfrc_applied[:]=[hinge,handle]
    for _ in range(round(seconds/model.opt.timestep)):
        mujoco.mj_step(model,data)


@pytest.mark.parametrize('heading', [0.,np.pi/2])
def test_latch_blocks_opens_closes_and_relatches_with_unchanged_collision_masks(heading):
    spec=mujoco.MjSpec();spec.compiler.degree=False;spec.option.timestep=.002
    install_latched_leaf(spec,'door',[0,0,0],heading,.9,2.1)
    model=spec.compile();data=mujoco.MjData(model)
    masks=(model.geom_contype.copy(),model.geom_conaffinity.copy())
    assert model.nu==0  # Test forces are external loads, not a robot demo.
    step(model,data,2.,5.,0.)
    assert 0 < data.qpos[0] < np.radians(2)
    assert any(model.geom(c.geom1).name=='door_strike' or model.geom(c.geom2).name=='door_strike'
               for c in data.contact)
    step(model,data,.5,-5.,0.)
    step(model,data,1.,-5.,2.5)
    assert data.qpos[1]>np.radians(50)
    step(model,data,3.,5.,2.5)
    assert data.qpos[0]>np.radians(85)
    step(model,data,4.,-5.,2.5)
    assert abs(data.qpos[0])<np.radians(1)
    step(model,data,2.,-2.,0.)
    assert abs(data.qpos[1])<np.radians(1)
    step(model,data,2.,5.,0.)
    assert data.qpos[0]<np.radians(2)
    np.testing.assert_array_equal(model.geom_contype,masks[0])
    np.testing.assert_array_equal(model.geom_conaffinity,masks[1])


@pytest.mark.parametrize('state',['open','closed'])
def test_latched_door_fits_existing_room_frame(state):
    spec=scene()
    records=install_room_doors(spec,{'room_door_0':state},latched=True)
    model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    assert model.njnt==2  # One leaf hinge and one handle hinge.
    assert records[0]['mechanism']=='passive_rotating_cam_latch'
    assert records[0]['robot_skill_validated'] is False
    assert all(c.dist>=-1e-6 for c in data.contact)
