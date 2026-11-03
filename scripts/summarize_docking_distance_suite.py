"""Summarize every configured case, preserving failures and incomplete runs."""
import argparse,json,math
from pathlib import Path
import numpy as np


def summarize(root):
    manifest=json.loads((root/'manifest.json').read_text());rows=[]
    for case in manifest['cases']:
        folder=root/case['name'];report=folder/'report.json'
        row=dict(name=case['name'],backoff=case['backoff'],initial_yaw_deg=math.degrees(case['yaw']),status='pending')
        if report.exists():
            r=json.loads(report.read_text());events=r.get('docking_events',[])
            alignment=next((e for e in reversed(events) if e['event']=='final_alignment'),{})
            pickup=next((e for e in reversed(events) if e['event']=='pickup_alignment'),{})
            selected=r.get('selected_object') or {};targets=r.get('targets',[])
            match=next((t for t in targets if t.get('native_id')==selected.get('native_id') and t.get('native_id') is not None),None)
            first=None
            if match is not None:
                for line in (folder/'progress.jsonl').read_text().splitlines():
                    progress=json.loads(line)
                    first=next((t for t in progress['targets'] if t['id']==match['track_id']),None)
                    if first:break
            distance=float(np.linalg.norm(first['position'][:2])) if first else None
            row.update(status=r['status'],docking_success=bool(r.get('docking_success')),contacts=len(r.get('external_contact_events',[])),
                       initial_distance_estimate_m=distance,simulated_seconds=r['simulated_seconds'],
                       final_distance_m=(r.get('readiness') or {}).get('distance_m'),
                       facing_error_deg=math.degrees(alignment['heading_error_rad']) if 'heading_error_rad' in alignment else None,
                       planned_lateral_offset_m=pickup.get('lateral_offset_m'),video=str(folder/'navigation.mp4'),
                       clearance_replans=sum(e['event']=='clearance_replan' for e in events),
                       rolling_advances=sum(e['event']=='rolling_advance' for e in events),
                       manipulation_ready=r.get('manipulation_ready',False))
        elif (root/(case['name']+'_result.json')).exists():
            row.update(json.loads((root/(case['name']+'_result.json')).read_text()))
        rows.append(row)
    measurements=root/'counter_alignment.json'
    alignment_by_case={r['case']:r for r in json.loads(measurements.read_text())} if measurements.exists() else {}
    for row in rows:
        alignment=alignment_by_case.get(row['name'],{})
        row['counter_normal_error_deg']=alignment.get('normal_error_deg')
        row['frontal_docking_diagnostic_pass']=bool(row.get('docking_success') and row.get('contacts',0)==0 and alignment.get('diagnostic_frontal_within_5deg',False))
    summary=dict(cases=rows,completed=sum(r['status']!='pending' for r in rows),
                 docking_passes=sum(bool(r.get('docking_success')) and r.get('contacts',0)==0 for r in rows),
                 frontal_docking_diagnostic_passes=sum(r['frontal_docking_diagnostic_pass'] for r in rows),
                 total=len(rows),scope='One scene, fresh resets, sensor-only search/approach/docking; no grasp tests.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    def fmt(v):return '—' if v is None else f'{v:.2f}'
    lines=['# Docking distance suite','',summary['scope'],'',
           'Success below means docking with no audited contacts. It is not grasp/task success.',
           '`docked_reachability_unverified` needs arm-path/grasp validation; `docked_fresh_confirmation_required` additionally needs fresh visual reconfirmation before manipulation.',
           'Distances are SLAM/visual estimates; the initial estimate is the first selected-object position relative to the SLAM origin.',
           '', 'Counter-normal error is an offline depth-plane estimate; ≤5° is the frontal-docking diagnostic, not an existing task-success criterion.',
           '', '| Case | Start (m) | Seconds | Final (m) | Target facing (°) | Counter normal (°) | Contacts | Status | Video |',
           '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for r in rows:
        link=f"[video]({r['name']}/navigation.mp4)" if r.get('video') else '—'
        lines.append(f"| {r['name']} | {fmt(r.get('initial_distance_estimate_m'))} | {fmt(r.get('simulated_seconds'))} | {fmt(r.get('final_distance_m'))} | {fmt(r.get('facing_error_deg'))} | {fmt(r.get('counter_normal_error_deg'))} | {r.get('contacts','—')} | {r['status']} | {link} |")
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args();summarize(a.output)
