"""Render saved physical states after execution (including failed attempts)."""
import argparse,json
from cross_episode_sim.recording.deferred import render_recording

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',required=True);p.add_argument('--output',required=True)
    p.add_argument('--diagnostic-output')
    a=p.parse_args();print(json.dumps(render_recording(a.recording,a.output,diagnostic_output=a.diagnostic_output),indent=2))
