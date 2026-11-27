"""Run any household recipe through the Qwen skill facade from initial state."""
import argparse,json
from pathlib import Path
from cross_episode_sim.evaluation.sensor_household import SensorHouseholdPipeline


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
    p.add_argument('--text-memory',type=Path,help='text memory root (task_language_memory_v4), or auto for this episode\'s entry in memory/episode_memory.json: the history procedure and '
        "this house's remembered object sources go into the Qwen plan and replan prompts (evaluation/text_memory.py)")
    p.add_argument('--visual-memory',type=Path,help='visual memory root, merged map folder, or auto (default none): exploration\'s first '
        'full turn matches onboard views against it and logs the robot pose in the memory frame; motion is unchanged')
    p.add_argument('--memory-navigation',action='store_true',help='same-layout episodes only: relocalize against the house tour map during '
        'exploration\'s first turn; when confident (fit <= 3 cm), navigation calls first move with the memory index (remembered routes '
        'or A* on the tour map) to a remembered manipulation start at the location, then the usual stages continue from there')
    p.add_argument('--memory-skip-explore',action='store_true',help='with --memory-navigation: a confident first-turn relocalization replaces exploration (memory tour map + semantic memory from the memory index)')
    p.add_argument('--memory-index',type=Path,help='memory index folder (default: <memory>/visual/memory_index_dev2)')
    a=p.parse_args()
    memory_index=None
    if a.memory_navigation or a.text_memory is not None or a.visual_memory is not None:
        # Memory folders: <dataset>/memory above this episode, or MEMORY_ROOT (memory_paths.py). Fixed here for
        # the whole process, before any memory module reads them.
        from cross_episode_sim.evaluation import memory_paths
        print('MEMORY_ROOT',memory_paths.use_episode(a.source),flush=True)
    if a.memory_navigation:
        from cross_episode_sim.evaluation.episode_memory import episode_memory
        if episode_memory(a.source) is None:
            print('memory navigation: no history for this episode (not a same-layout episode); running without it',flush=True)
        else:
            a.visual_memory=memory_paths.tour_root()
            memory_index=a.memory_index or memory_paths.index_dir()
    # 'auto': this episode's own memory from <v4>/memory/episode_memory.json (none for new layouts).
    from cross_episode_sim.evaluation.episode_memory import resolve
    a.text_memory=resolve(a.text_memory,a.source,'text_memory');a.visual_memory=resolve(a.visual_memory,a.source,'visual_memory')
    pipeline=SensorHouseholdPipeline(a.source,a.output,navigation_seconds=a.navigation_seconds,exploration_seconds=a.exploration_seconds,
        pointing_device='cuda:0',furniture_device='cuda:0',continue_on_failure=not a.fail_fast,
        qwen_interface=not a.legacy_m2t2_interface,planner=a.planner,
        wall_seconds_limit=a.wall_seconds_limit,sim_seconds_limit=a.sim_seconds_limit,max_calls=a.max_calls,text_memory=a.text_memory,visual_memory=a.visual_memory)
    pipeline.memory_index=memory_index;pipeline.memory_skip_explore=bool(a.memory_skip_explore and memory_index is not None)
    try:report=pipeline.run()
    finally:
        if not a.skip_video_render and a.output.exists():
            from cross_episode_sim.recording.deferred import render_episode
            render_episode(a.output)
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
