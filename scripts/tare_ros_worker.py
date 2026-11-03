"""Private ROS 1 worker. Run with .venv-tare, not the simulator environment."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import threading
import time
import xmlrpc.client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    def terminate(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    # Load only the transport module; ROS Python 3.9 must not import simulator
    # modules that require Python 3.11, MuJoCo, or Open3D.
    import importlib.util
    spec = importlib.util.spec_from_file_location('tare_transport', args.root/'cross_episode_sim/navigation/tare.py')
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    send_message, receive_message = transport.send_message, transport.receive_message
    import yaml
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1', 0))
        ros_port = reserve.getsockname()[1]
    os.environ.update(ROS_MASTER_URI=f'http://127.0.0.1:{ros_port}', ROS_IP='127.0.0.1',
                      ROS_HOME=str(args.output/'ros'), ROS_LOG_DIR=str(args.output/'ros/log'))
    upstream = args.root/'external/tare_planner'
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output(['git', '-C', str(upstream), 'diff', '--stat'], text=True)
    (args.output/'upstream.json').write_text(json.dumps(dict(repository='https://github.com/caochao39/tare_planner',
        revision=revision, tracked_source_changes=changes), indent=2))
    workspace = args.root/'.tools/tare_ws'
    os.environ['LD_LIBRARY_PATH'] = ':'.join([str(workspace/'devel/lib'),
        str(upstream/'src/tare_planner/or-tools/lib'), str(args.root/'.venv-tare/lib'),
        os.environ.get('LD_LIBRARY_PATH', '')])
    processes = []
    logs = []
    connection = None
    try:
        def launch(command, name):
            log = open(args.output/name, 'w'); logs.append(log)
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            return process
        master = launch(['roscore', '-p', str(ros_port)], 'roscore.log')
        for _ in range(200):
            try:
                xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/tare_probe')
                break
            except OSError:
                if master.poll() is not None:
                    raise RuntimeError('ROS master exited')
                time.sleep(.05)
        else:
            raise TimeoutError('ROS master startup')
        import rospy
        from std_msgs.msg import Header
        from sensor_msgs.msg import PointField
        from sensor_msgs import point_cloud2
        from nav_msgs.msg import Odometry, Path as ROSPath
        from geometry_msgs.msg import PointStamped
        from std_msgs.msg import Bool
        from rosgraph_msgs.msg import Clock
        rospy.init_node('onboard_tare_bridge', disable_signals=True)
        rospy.set_param('/use_sim_time', True)
        parameters = yaml.safe_load((upstream/'src/tare_planner/config/indoor.yaml').read_text())
        parameters.update(yaml.safe_load((args.root/'configs/navigation/tare_rgbd.yaml').read_text()))
        for key, value in parameters.items():
            rospy.set_param('/tare/tare_planner_node/'+key, value)
        (args.output/'parameters.json').write_text(json.dumps(parameters, indent=2))
        lock = threading.Lock()
        state = dict(path=[], global_path=[], waypoint=None, finished=False, sequence=0, path_time=None)
        def path_callback(msg):
            with lock:
                state.update(path=[[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses],
                             path_time=msg.header.stamp.to_sec(), sequence=state['sequence']+1)
        def waypoint_callback(msg):
            with lock:
                state['waypoint'] = [msg.point.x, msg.point.y, msg.point.z]
        def finish_callback(msg):
            with lock:
                state['finished'] = bool(msg.data)
        def global_callback(msg):
            with lock:
                state['global_path'] = [[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in msg.poses]
        # Upstream advertises exploration_path but its publish call is commented
        # out. local_path and global_path are the actual published planner paths.
        rospy.Subscriber('/tare/local_path', ROSPath, path_callback, queue_size=1)
        rospy.Subscriber('/tare/global_path', ROSPath, global_callback, queue_size=1)
        rospy.Subscriber('/way_point', PointStamped, waypoint_callback, queue_size=1)
        rospy.Subscriber('/tare/exploration_finish', Bool, finish_callback, queue_size=1)
        clock = rospy.Publisher('/clock', Clock, queue_size=2)
        odometry = rospy.Publisher('/state_estimation_at_scan', Odometry, queue_size=1)
        cloud = rospy.Publisher('/registered_scan', point_cloud2.PointCloud2, queue_size=1)
        command = [str(workspace/'devel/lib/tare_planner/tare_planner_node'), '__ns:=/tare']
        if os.environ.get('TARE_GDB') == '1':
            command = ['/usr/bin/gdb', '-batch', '-ex', 'run', '-ex', 'thread apply all bt', '--args']+command
        node = launch(command, 'planner.log')
        for _ in range(200):
            if node.poll() is not None:
                raise RuntimeError('Upstream TARE exited; see planner.log')
            if odometry.get_num_connections() and cloud.get_num_connections():
                break
            time.sleep(.05)
        else:
            raise TimeoutError('TARE subscriptions did not connect')
        fields = [PointField(name, index*4, PointField.FLOAT32, 1) for index, name in enumerate(('x', 'y', 'z'))]
        with socket.socket() as server:
            server.bind(('127.0.0.1', args.port)); server.listen(1)
            connection, _ = server.accept()
            with connection, open(args.output/'planner_responses.jsonl', 'w', buffering=1) as trace:
                send_message(connection, {'status': 'ready'})
                previous = -1.
                while True:
                    request = receive_message(connection)
                    if request.get('close'):
                        break
                    timestamp = request['timestamp']
                    if timestamp <= previous:
                        raise ValueError('Observation timestamps must increase')
                    previous = timestamp
                    stamp = rospy.Time.from_sec(timestamp)
                    msg = Odometry(); msg.header = Header(stamp=stamp, frame_id='map')
                    msg.child_frame_id = 'navigation_sensor'
                    msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = request['sensor_position']
                    msg.pose.pose.orientation.z = math.sin(request['yaw']/2)
                    msg.pose.pose.orientation.w = math.cos(request['yaw']/2)
                    odometry.publish(msg)
                    cloud.publish(point_cloud2.create_cloud(msg.header, fields, request['points']))
                    # Allow callbacks to consume observations before advancing ROS timers.
                    time.sleep(.01)
                    clock.publish(Clock(clock=stamp))
                    time.sleep(.02)
                    if node.poll() is not None:
                        raise RuntimeError(f'Upstream TARE exited during planning, code {node.returncode}; see planner.log')
                    with lock:
                        response = dict(state, observation_time=timestamp)
                    trace.write(json.dumps(response)+'\n')
                    send_message(connection, response)
    except Exception as exc:
        (args.output/'failure.json').write_text(json.dumps(dict(error=str(exc)), indent=2))
        if connection is not None:
            try:
                send_message(connection, {'error': str(exc)})
            except OSError:
                pass
        raise
    finally:
        for process in reversed(processes):
            process.terminate()
        for process in reversed(processes):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
