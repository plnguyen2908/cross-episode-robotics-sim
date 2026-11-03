"""Sensor-only transport to the unmodified upstream TARE ROS process.

No MuJoCo imports, room graph, reference trajectory, or simulator world poses.
The planner's position is the RGB-D optical origin, so occupancy rays originate
at the real sensor rather than at an invented lidar at the robot centre.
"""
from pathlib import Path
import json
import math
import socket
import struct
import subprocess
import time
import numpy as np


def observation_payload(camera, tracking, stride=6, max_depth=6.):
    if not tracking.accepted:
        raise ValueError('TARE requires accepted visual tracking')
    pose = np.asarray(tracking.T_map_base)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('Invalid estimated pose')
    yy, xx = np.mgrid[0:camera.depth_m.shape[0]:stride, 0:camera.depth_m.shape[1]:stride]
    depth = camera.depth_m[::stride, ::stride]
    good = camera.valid_depth[::stride, ::stride] & np.isfinite(depth) & (depth > .1) & (depth <= max_depth)
    if camera.robot_mask is not None:
        good &= ~camera.robot_mask[::stride, ::stride]
    z = depth[good]
    points = np.column_stack(((xx[good]-camera.K[0, 2])*z/camera.K[0, 0],
                              (yy[good]-camera.K[1, 2])*z/camera.K[1, 1], z))
    optical = pose @ camera.T_base_optical
    points = points @ optical[:3, :3].T + optical[:3, 3]
    yaw = math.atan2(pose[1, 0], pose[0, 0])
    return dict(timestamp=float(tracking.timestamp), points=points.tolist(),
                sensor_position=optical[:3, 3].tolist(), yaw=yaw)


def send_message(sock, message):
    raw = json.dumps(message, allow_nan=False, separators=(',', ':')).encode()
    sock.sendall(struct.pack('!I', len(raw))+raw)


def receive_message(sock):
    def exact(n):
        result = bytearray()
        while len(result) < n:
            chunk = sock.recv(n-len(result))
            if not chunk:
                raise EOFError('TARE worker disconnected')
            result.extend(chunk)
        return result
    size = struct.unpack('!I', exact(4))[0]
    if size > 32*1024*1024:
        raise ValueError('Oversized TARE message')
    return json.loads(exact(size))


class TarePlanner:
    """Own a private ROS master + upstream planner; update returns paths, not actions."""
    def __init__(self, output, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[2])
        if not (self.root/'.tools/tare_ws/devel/lib/tare_planner/tare_planner_node').is_file():
            raise FileNotFoundError('Build upstream TARE first: bash scripts/setup_tare.sh')
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.process = None
        self.socket = None
        self.log = open(self.output/'worker.log', 'w')
        with socket.socket() as reserve:
            reserve.bind(('127.0.0.1', 0))
            port = reserve.getsockname()[1]
        command = [str(self.root/'.tools/bin/micromamba'), 'run', '-p', str(self.root/'.venv-tare'),
                   'python', str(self.root/'scripts/tare_ros_worker.py'), '--port', str(port),
                   '--output', str(self.output), '--root', str(self.root)]
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+60
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f'TARE worker failed: see {self.output}/worker.log')
                try:
                    self.socket = socket.create_connection(('127.0.0.1', port), timeout=1)
                    self.socket.settimeout(30)
                    ready = receive_message(self.socket)
                    if ready.get('status') != 'ready':
                        raise RuntimeError(str(ready))
                    break
                except ConnectionRefusedError:
                    time.sleep(.1)
            else:
                raise TimeoutError('TARE worker startup timed out')
        except BaseException:
            self.close()
            raise

    def update(self, camera, tracking):
        send_message(self.socket, observation_payload(camera, tracking))
        try:
            result = receive_message(self.socket)
        except EOFError as exc:
            failure = self.output/'failure.json'
            if failure.exists():
                raise RuntimeError(json.loads(failure.read_text())['error']) from exc
            raise
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def close(self):
        if self.socket:
            try:
                send_message(self.socket, {'close': True})
            except (OSError, EOFError):
                pass
            self.socket.close()
            self.socket = None
        if self.process:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
            self.process = None
        self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
