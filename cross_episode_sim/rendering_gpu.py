"""Match CUDA and EGL devices by physical PCI address, not enumeration order."""

import json
import os
import subprocess
import sys
from pathlib import Path


def pci_key(address):
    domain, bus, slot = address.strip().lower().split(':')
    return int(domain, 16), int(bus, 16), slot


def match_devices(requested, nvidia, egl):
    result = {}
    for requested_gpu in requested:
        matches = [row for row in nvidia if requested_gpu == row['index'] or
                   (requested_gpu.startswith('GPU-') and row['uuid'].startswith(requested_gpu))]
        if len(matches) != 1:
            raise ValueError(f'GPU {requested_gpu!r} does not identify one NVIDIA GPU')
        gpu = matches[0]
        device = next((row for row in egl if pci_key(row['pci']) == pci_key(gpu['pci'])), None)
        if device is None:
            raise RuntimeError(f'No EGL rendering device matches GPU {requested_gpu} ({gpu["pci"]})')
        result[requested_gpu] = dict(cuda_uuid=gpu['uuid'], egl_device=device['index'])
    return result


def resolve_gpus(requested):
    listing = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id', '--format=csv,noheader'],
        text=True, timeout=20)
    nvidia = [dict(zip(('index', 'uuid', 'pci'), map(str.strip, line.split(','))))
              for line in listing.splitlines() if line.strip()]
    # Keep EGL imports and their process-global display selection out of the
    # launcher. Device enumeration must see all GPUs, including inherited masks.
    env = {**os.environ, 'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl'}
    env.pop('CUDA_VISIBLE_DEVICES', None)
    devices = subprocess.check_output(
        [sys.executable, '-m', 'cross_episode_sim.rendering_gpu'],
        env=env, text=True, timeout=30)
    return match_devices(requested, nvidia, json.loads(devices))


def worker_environment(env, gpu, binding=None):
    result = {**env, 'CUDA_VISIBLE_DEVICES': binding['cuda_uuid'] if binding else gpu}
    if binding is not None:
        result['MUJOCO_EGL_DEVICE_ID'] = str(binding['egl_device'])
    return result


def egl_devices():
    import ctypes

    from mujoco.egl import egl_ext as egl

    query = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(
        egl.eglGetProcAddress('eglQueryDeviceStringEXT'))
    devices = []
    for index, device in enumerate(egl.eglQueryDevicesEXT(max_devices=256)):
        extensions = query(device, 0x3055) or b''  # EGL_EXTENSIONS
        if b'EGL_NV_device_cuda' not in extensions.split():
            continue
        node = query(device, 0x3233)  # EGL_DRM_DEVICE_FILE_EXT
        if node:
            physical = Path('/sys/class/drm') / Path(node.decode()).name / 'device'
            if physical.exists():
                devices.append(dict(index=index, pci=physical.resolve().name))
    return devices


if __name__ == '__main__':
    print(json.dumps(egl_devices()))
