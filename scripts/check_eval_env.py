"""Self-check for the evaluation venvs made by scripts/setup_eval_env.sh.

    source eval_env.sh
    .venv-sift/bin/python      scripts/check_eval_env.py sift
    .venv-m2t2/bin/python      scripts/check_eval_env.py m2t2
    .venv-graspgenx/bin/python scripts/check_eval_env.py graspgenx

Each check runs real CUDA work on the first visible GPU (CUDA_VISIBLE_DEVICES).
Exits non-zero and prints the error text of every failed check.
"""
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results = []


def check(name):
    def wrap(fn):
        t = time.time()
        try:
            detail = fn()
            results.append((name, True, detail))
            print(f'PASS {name}: {detail} ({time.time()-t:.1f}s)', flush=True)
        except Exception as exc:  # report the real error text and keep going
            results.append((name, False, f'{type(exc).__name__}: {exc}'))
            print(f'FAIL {name}: {type(exc).__name__}: {exc}', flush=True)
            traceback.print_exc()
        return fn
    return wrap


def torch_cuda():
    import torch
    assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
    x = torch.randn(512, 512, device='cuda')
    float((x @ x).sum())
    return f'torch {torch.__version__} cuda {torch.version.cuda} on {torch.cuda.get_device_name(0)}'


def sift():
    check('torch CUDA')(torch_cuda)

    @check('mujoco EGL render')
    def _():
        assert os.environ.get('MUJOCO_GL') == 'egl', 'set MUJOCO_GL=egl (source eval_env.sh)'
        import mujoco
        import numpy as np
        m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos="0 0 3"/>'
                                           '<geom type="box" size=".2 .2 .2" rgba="1 0 0 1"/></worldbody></mujoco>')
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        with mujoco.Renderer(m, 96, 128) as r:
            r.update_scene(d); rgb = r.render()
            r.enable_depth_rendering(); r.update_scene(d); depth = r.render()
        assert rgb[..., 0].max() > 100 and np.isfinite(depth).all(), 'blank EGL frame'
        return f'mujoco {mujoco.__version__}, rgb max {rgb.max()}, depth range {depth.min():.2f}-{depth.max():.2f}'

    @check('open3d CUDA')
    def _():
        import open3d as o3d
        assert o3d.core.cuda.is_available(), 'open3d.core.cuda.is_available() is False'
        t = o3d.core.Tensor.ones((256, 256), o3d.core.float32, o3d.core.Device('CUDA:0'))
        assert float((t @ t).sum().item()) == 256 ** 3
        return f'open3d {o3d.__version__}, {o3d.core.cuda.device_count()} CUDA device(s)'

    @check('pycolmap CUDA')
    def _():
        import pycolmap
        assert pycolmap.has_cuda, 'pycolmap.has_cuda is False'
        return f'pycolmap {pycolmap.__version__} has_cuda'

    @check('cuRobo IK (NVRTC kernels)')
    def _():
        import torch
        from curobo._src.solver.solver_ik import IKSolver
        from curobo._src.solver.solver_ik_cfg import IKSolverCfg
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo._src.types.pose import Pose
        from curobo._src.types.tool_pose import GoalToolPose
        dev = DeviceCfg(device=torch.device('cuda:0'), dtype=torch.float32)
        cfg = IKSolverCfg.create(robot='franka.yml', device_cfg=dev, num_seeds=32, use_cuda_graph=False)
        cfg.use_lm_seed = False
        ik = IKSolver(cfg)
        goal = Pose(position=torch.tensor([[0.4, 0.0, 0.4]], **dev.as_torch_dict()),
                    quaternion=torch.tensor([[0.0, 1.0, 0.0, 0.0]], **dev.as_torch_dict()))
        frames = ik.kinematics.tool_frames
        res = ik.solve_pose(goal_tool_poses=GoalToolPose.from_poses({frames[0]: goal.unsqueeze(1)}, ordered_tool_frames=frames))
        ok = bool(res.success.any())
        assert ok, f'IK failed, position error {res.position_error}'
        from importlib.metadata import version
        return f'nvidia-curobo {version("nvidia-curobo")}, franka IK success'

    @check('pipeline imports')
    def _():
        import molmo_spaces, daaam, transformers, cv2  # noqa: F401
        from cross_episode_sim.evaluation import sensor_household  # noqa: F401
        from cross_episode_sim.evaluation.articulated_scene import load_saved_scene  # noqa: F401
        from cross_episode_sim.perception import daaam_grounding, molmo_pointing  # noqa: F401
        from daaam.segmentation.services import SegmentationService  # noqa: F401
        return (f'molmo_spaces from {Path(molmo_spaces.__file__).parent}, transformers {transformers.__version__}, '
                f'cv2 {cv2.__version__}')

    @check('FastSAM checkpoint')
    def _():
        from ultralytics import FastSAM
        import numpy as np
        ckpt = os.environ.get('FASTSAM_CHECKPOINT', str(ROOT/'external/DAAAM/checkpoints/fastsam/FastSAM-s.pt'))
        out = FastSAM(ckpt)(np.zeros((256, 256, 3), np.uint8), device='cuda:0', verbose=False)
        return f'FastSAM-s ran on cuda:0 ({len(out)} result)'


def m2t2():
    check('torch CUDA')(torch_cuda)

    @check('pointnet2_ops CUDA op')
    def _():
        import torch
        from pointnet2_ops import pointnet2_utils
        idx = pointnet2_utils.furthest_point_sample(torch.rand(1, 2048, 3, device='cuda'), 64)
        assert idx.shape == (1, 64)
        return 'furthest_point_sample on cuda'

    @check('M2T2 model + checkpoint + inference')
    def _():
        import numpy as np
        os.environ.pop('SENSOR_MODEL_ENDPOINTS', None)
        sys.path.insert(0, str(ROOT))
        from cross_episode_sim.control.m2t2_policy import M2T2Policy
        ckpt = os.environ['M2T2_CHECKPOINT']
        policy = M2T2Policy(ckpt, str(ROOT/'external/M2T2'), device='cuda:0')
        rng = np.random.default_rng(0)
        table = np.c_[rng.uniform(.3, .9, 6000), rng.uniform(-.3, .3, 6000), np.zeros(6000)]
        box = np.c_[rng.uniform(.55, .62, 3000), rng.uniform(-.03, .03, 3000), rng.uniform(0, .12, 3000)]
        pts = np.r_[table, box].astype(np.float32)
        out = policy.predict(pts, np.full_like(pts, .5))
        return f'checkpoint {Path(ckpt).name}, predict keys {sorted(out)[:6] if isinstance(out, dict) else type(out).__name__}'


def graspgenx():
    check('torch CUDA')(torch_cuda)

    @check('GraspGen-X model + checkpoints + inference')
    def _():
        import numpy as np
        sys.path.insert(0, str(ROOT))
        os.chdir(ROOT)
        from graspgenx import get_checkpoints_version_dir
        from cross_episode_sim.control.graspgenx_policy import GraspGenXPolicy
        policy = GraspGenXPolicy()
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 2*np.pi, 3000); z = rng.uniform(0, .1, 3000)
        cup = np.c_[.04*np.cos(a), .04*np.sin(a), z] + [.5, 0, .8]
        out = policy.predict(cup.astype(np.float32), seed=0)
        n = len(out['grasps'])
        assert n > 0, 'no grasps returned'
        return f'{n} grasps, checkpoints {get_checkpoints_version_dir()}'


if __name__ == '__main__':
    kind = sys.argv[1] if len(sys.argv) > 1 else 'sift'
    print(f'== check {kind}: python {sys.version.split()[0]} at {sys.executable}', flush=True)
    {'sift': sift, 'm2t2': m2t2, 'graspgenx': graspgenx}[kind]()
    failed = [r for r in results if not r[1]]
    print(f'== {kind}: {len(results)-len(failed)}/{len(results)} checks passed', flush=True)
    sys.exit(1 if failed else 0)
