"""Author one initial RoboCasa scene for the shared cuRobo runner; no grasp execution."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
import robosuite
from cross_episode_sim.grasping.grasp_test import TidyBotGraspTest
from cross_episode_sim.grasping.robosuite_tidybot import controller_config
from cross_episode_sim.manipulation.molmo_objects import register, OUTPUT, convert, assets_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--source', choices=('molmo', 'robocasa'), default='molmo')
    args = p.parse_args()
    if args.source == 'molmo':
        if not (OUTPUT/args.asset/'metadata.json').exists():
            convert(args.asset, assets_root())
        register([args.asset])
        TidyBotGraspTest.test_asset = str((OUTPUT / args.asset).resolve())
    else:
        TidyBotGraspTest.test_asset = args.asset
    TidyBotGraspTest.trial_seed = args.seed
    env = robosuite.make('TidyBotGraspTest', robots='TidyBotFranka',
        controller_configs=controller_config(), layout_ids=[1101], style_ids=[1],
        seed=7, initialization_noise=None, use_camera_obs=False,
        has_renderer=False, has_offscreen_renderer=False, randomize_cameras=False)
    try:
        env.reset()
        env.sim.forward()
        model, data = env.sim.model._model, env.sim.data._data
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, state, spec)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'scene.xml').write_text(env.model.get_xml())
        np.savez_compressed(args.output / 'states.npz', states=[state], labels=['authored initial state'])
        obj = env.objects['test_object']
        (args.output / 'setup.json').write_text(json.dumps(dict(asset=args.asset,
            source=args.source, model_xml=str(Path(obj.mjcf_path).resolve()),
            object_scale=np.asarray(obj._scale).tolist(),
            seed=args.seed, physics_executed=False, grasp_executed=False), indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
