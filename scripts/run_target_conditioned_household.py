"""Run any household recipe through the Qwen skill facade from initial state."""
import argparse,json
from pathlib import Path
from cross_episode_sim.evaluation.target_conditioned_household import TargetConditionedHouseholdPipeline as SensorHouseholdPipeline


def main():
    import sys
    from cross_episode_sim.runtime.logging import TimestampedStream
    sys.stdout=TimestampedStream(sys.stdout);sys.stderr=TimestampedStream(sys.stderr)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--navigation-seconds',type=float,default=180.)
    p.add_argument('--exploration-seconds',type=float,default=900.,help='Upper limit for house exploration only; it stops as soon as the stations are found')
    p.add_argument('--skip-video-render',action='store_true')
    p.add_argument('--legacy-m2t2-interface',action='store_true',help='Retain the prior M2T2-only dispatch, including the qualified set-table baseline')
    p.add_argument('--fail-fast',action='store_true',help='Legacy abort on first failed household skill; default attempts the full recipe.')
    p.add_argument('--planner',choices=('recipe','qwen_zero_shot','qwen_zero_shot_pure','qwen_replan','gpt_zero_shot','gpt_replan'),default='recipe',
        help='recipe: the fixed semantic recipe; qwen_zero_shot: Qwen writes the whole plan from the instruction with the '
             'planning-contract rules and up to 2 repair requests, executed open loop; qwen_zero_shot_pure: one request, no rules; '
             'qwen_replan: as qwen_zero_shot, and after a failed call Qwen rewrites the rest of the plan from public outcomes; '
             'gpt_zero_shot / gpt_replan: the same with GPT (OpenAI API, PLANNER_MODEL, default gpt-6.1-sol)')
    p.add_argument('--wall-seconds-limit',type=float,help='stop at this wall time from episode start and score the live state (GPT eval: 2700)')
    p.add_argument('--sim-seconds-limit',type=float,help='stop at this simulated time and score the live state (GPT eval: 900)')
    p.add_argument('--max-calls',type=int,help='run at most this many plan calls (GPT eval: 100)')
    a=p.parse_args()
    pipeline=SensorHouseholdPipeline(a.source,a.output,navigation_seconds=a.navigation_seconds,exploration_seconds=a.exploration_seconds,
        pointing_device='cuda:0',furniture_device='cuda:0',continue_on_failure=not a.fail_fast,
        qwen_interface=not a.legacy_m2t2_interface,planner=a.planner,
        wall_seconds_limit=a.wall_seconds_limit,sim_seconds_limit=a.sim_seconds_limit,max_calls=a.max_calls)
    try:report=pipeline.run()
    finally:
        if not a.skip_video_render and a.output.exists():
            from cross_episode_sim.recording.deferred import render_episode
            render_episode(a.output)
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
