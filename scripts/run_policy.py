"""Roll out a policy in a task environment and record the result.

    python scripts/run_policy.py --task breakfast --policy examples.replay_policy:ReplayPolicy \
        --policy-arg dataset=datasets/breakfast.hdf5 --episodes 1 --output runs/policy_eval

A policy is any class with `act(observation) -> action` and an optional
`reset()`; see `cross_episode_sim.env.task_env` for the observation and action
layout. The script writes one directory per episode with a video, the step
count, the final goal predicates and any physics failure.
"""

import argparse
import importlib
import json
from pathlib import Path

import imageio.v2 as imageio

from cross_episode_sim.env.task_env import TaskEnv


def load_policy(spec, arguments):
    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)(**arguments)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="breakfast")
    parser.add_argument("--policy", required=True, help="module:Class")
    parser.add_argument("--policy-arg", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--camera", action="append", dest="cameras")
    parser.add_argument("--max-steps", type=int, default=25 * 60 * 30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = load_policy(args.policy, dict(item.split("=", 1) for item in args.policy_arg))
    cameras = tuple(args.cameras or ("robot_0/wrist_cam",))
    env = TaskEnv(args.task, output_root=args.output / "episodes", cameras=cameras, max_steps=args.max_steps)
    results = []
    for episode in range(args.episodes):
        observation, info = env.reset(seed=args.seed + episode)
        if hasattr(policy, "reset"):
            policy.reset()
        directory = args.output / f"rollout_{episode:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(directory / "rollout.mp4", fps=env.metadata["render_fps"]) as video:
            terminated = truncated = False
            steps = 0
            while not (terminated or truncated):
                observation, reward, terminated, truncated, info = env.step(policy.act(observation))
                video.append_data(env.render())
                steps += 1
        result = dict(episode=episode, seed=args.seed + episode, steps=steps, success=reward > 0,
                      goals=info.get("goals"), failure=info.get("failure"))
        (directory / "result.json").write_text(json.dumps(result, indent=2))
        results.append(result)
        print(json.dumps(result), flush=True)
    env.close()
    rate = sum(r["success"] for r in results) / len(results)
    (args.output / "summary.json").write_text(json.dumps(dict(success_rate=rate, episodes=results), indent=2))
    print(f"success rate {rate:.2f} over {len(results)} episodes")


if __name__ == "__main__":
    main()
