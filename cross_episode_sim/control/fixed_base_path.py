import numpy as np,time
from cross_episode_sim.control.observed_grasp_planner import ObservedGraspPlanner

class FixedBasePath(ObservedGraspPlanner):
 def planning_boxes(self):
  result=[];removed=[]
  for box in self.boxes:
   center=np.asarray(box.pose[:3]);half=np.asarray(box.dims)/2
   support_floor=center[2]+half[2]<=.03 and center[2]-half[2]>=-.06
   if support_floor:
    for sphere in self.fixed_floor_support_spheres:
     d=np.maximum(abs(sphere[:3]-center)-half,0)
     if np.linalg.norm(d)<sphere[3]+.005:
      removed.append(box);break
    else:result.append(box)
   else:result.append(box)
  print('FIXED_BASE_SUPPORT_CELLS',len(removed),'all other obstacles retained',flush=True)
  return result
 def _graph_joint_goal(self,q,goal,start):
  import torch
  from curobo.types import JointState
  from cross_episode_sim.control.curobo_scene import load_observed_cuboids
  def state(x):return JointState.from_position(torch.tensor(np.asarray(x)[None],device='cuda',dtype=torch.float32),joint_names=self.arm.names)
  load_observed_cuboids(self.arm.planner,self.planning_boxes())
  print('NATIVE_FILTERED_START',self.arm.planner.graph_planner.check_samples_feasibility(torch.tensor(q[None],device='cuda',dtype=torch.float32)),flush=True)
  result=self.cspace_search(state(goal),state(q))
  if result is None or not result.success.any():raise RuntimeError('No fixed-base cspace path')
  b,s=result.success.nonzero(as_tuple=True);end=int(result.interpolated_last_tstep[b[0],s[0]].item());tr=result.interpolated_trajectory
  indices=[tr.joint_names.index(n) for n in self.arm.names];raw=tr.position[b[0],s[0],:end,indices].cpu().numpy()
  old=np.arange(len(raw))*self.arm.dt;times=np.arange(0,old[-1]+.019,.02)
  path=np.column_stack([np.interp(times,old,raw[:,j]) for j in range(7)])
  # Original observed world, unchanged full-body spheres and floor-contact rule.
  if not self.clear(path):raise RuntimeError('Full-world validation failed: '+str(self.last_rejection))
  self.events.append(dict(method='fixed_base_cspace_full_world_validated',wall_seconds=time.monotonic()-start,simulated_seconds=(len(path)-1)*self.dt))
  return path
