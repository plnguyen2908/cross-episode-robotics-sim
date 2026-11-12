"""Point at a receptacle for docking; no support fit, grasp or placement IK."""
import argparse,json
from pathlib import Path
import numpy as np
from cross_episode_sim.runtime import artifacts
from cross_episode_sim.runtime.episode import load_checkpoint,episode_sensors
from cross_episode_sim.navigation.shaped_docking import RobotGeometry
from cross_episode_sim.control.observed_payload import ObservedPayload
from cross_episode_sim.control.sensor_appliance_backend import camera_points
from cross_episode_sim.perception.molmo_pointing import MolmoPointer
from cross_episode_sim.perception.approach_target import locate_approach_target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dock',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--receptacle',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    model,data=load_checkpoint(a.dock)
    geometry=RobotGeometry(model)
    tool=artifacts.load_array(a.dock/'observed_payload_tool.npy')
    with episode_sensors(model,data) as sensors:
        raw=sensors.capture()
        packet=ObservedPayload(geometry,tool).filter_packet(raw)
        try:target,evidence=locate_approach_target(packet,MolmoPointer(),a.receptacle)
        except ValueError as exc:
            artifacts.write_text(a.output/'report.json',json.dumps(dict(
                observation_success=False,error=str(exc),approach_anchor_only=True)))
            return 1
        clouds=[camera_points(camera)[:2] for camera in packet.cameras.values()]
        points=np.concatenate([v[0] for v in clouds]);rgb=np.concatenate([v[1] for v in clouds])
        from cross_episode_sim.perception.receptacle_front import observed_front
        evidence['observed_front']=observed_front(points,target)
        keep=np.linalg.norm(points,axis=1)<6.
        tcp=geometry.tcp_pose(packet.arm.position);held=tool@tcp[:3,:3].T+tcp[:3,3]
        radius=float(np.linalg.norm(held[:,:2]-np.median(held[:,:2],axis=0),axis=1).max()+.03)
        # Legacy docking schema: table_surface here contains only the observed
        # anchor. approach_anchor_only forbids using it as release geometry.
        artifacts.save_archive(a.output/'input.npz',points=points[keep],rgb=rgb[keep],
            held_points=held,ee_pose=tcp,table_surface=target[None],preferred_contacts=target[None])
        artifacts.save_array(a.output/'observed_payload_tool.npy',tool)
        artifacts.write_text(a.output/'support_filter.json',json.dumps(dict(radius_m=radius)))
        artifacts.write_text(a.output/'report.json',json.dumps(dict(observation_success=True,
            semantic_receptacle=a.receptacle,**evidence)))
        print('APPROACH_TARGET',json.dumps(dict(position_base=target.tolist(),**evidence)),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
