"""Offline arm/base/wait smoke checks on saved scenes; not task qualification."""
from pathlib import Path
import json,tempfile
import numpy as np
from cross_episode_sim.agent_policy.environment import Environment
root=Path(__file__).resolve().parents[1];out=Path(tempfile.mkdtemp(prefix='astra_physics_',dir=root/'runs'))
report=[]
for task,src in json.loads((root/'configs/vla/controller_demo_sources.json').read_text()).items():
 d=out/task;d.mkdir();e=Environment(root/src,d,root/'configs/controllers/molmospaces_mobile_franka.json')
 try:
  q=e.state()['arm_joints']; q[0]+=.01
  e.act('move_joints',{'arm':q,'gripper':0,'seconds':.5})
  e.act('move_base',{'forward':.01,'left':0,'yaw':.01,'seconds':.5})
  e.act('wait',{'seconds':.2})
  state,images=e.observe(0)
  assert state['simulation_seconds']>1.1
  assert all(p.exists() for p in images)
  report.append({'task':task,'seconds':state['simulation_seconds'],'goals':e.goals(),'arm_motion_error':float(np.max(abs(np.array(state['arm_joints'])-q)))})
 finally:e.close()
 print(report[-1],flush=True)
(out/'report.json').write_text(json.dumps(report,indent=2));print(out)
