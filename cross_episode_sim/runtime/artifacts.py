"""Explicit in-memory stage handoffs for an active episode.

Legacy standalone tools can still read/write files. Nothing monkey-patches
NumPy, pathlib, or MuJoCo. Paths are logical artifact keys during an episode.
"""
from pathlib import Path
import numpy as np


def owner(path):
    from .episode import current_runtime
    runtime=current_runtime()
    if runtime is None or runtime.artifact_root is None:return None
    try:Path(path).resolve().relative_to(runtime.artifact_root)
    except (ValueError,TypeError):return None
    return runtime


def key(path):return str(Path(path).resolve())


def exists(path):
    runtime=owner(path)
    return (runtime is not None and key(path) in runtime.artifacts) or Path(path).exists()


def read_text(path,*args,**kwargs):
    runtime=owner(path)
    if runtime is not None and key(path) in runtime.artifacts:return runtime.artifacts[key(path)]
    return Path(path).read_text(*args,**kwargs)


def write_text(path,text,*args,**kwargs):
    runtime=owner(path)
    if runtime is None:return Path(path).write_text(text,*args,**kwargs)
    runtime.artifacts[key(path)]=text
    return len(text)


def load_array(path,*args,**kwargs):
    runtime=owner(path)
    if runtime is not None and key(path) in runtime.artifacts:
        value=runtime.artifacts[key(path)]
        return {k:v.copy() for k,v in value.items()} if isinstance(value,dict) else value.copy()
    return np.load(path,*args,**kwargs)


def save_array(path,array,*args,**kwargs):
    runtime=owner(path)
    if runtime is None:return np.save(path,array,*args,**kwargs)
    runtime.artifacts[key(path)]=np.array(array,copy=True)


def save_archive(path,*args,**kwargs):
    runtime=owner(path)
    if runtime is None:return np.savez_compressed(path,*args,**kwargs)
    runtime.artifacts[key(path)]={**{f'arr_{i}':np.array(v,copy=True) for i,v in enumerate(args)},**{k:np.array(v,copy=True) for k,v in kwargs.items()}}


def save_model(model,path,buffer=None):
    runtime=owner(path)
    if runtime is not None:
        runtime.artifacts[key(path)]='live episode model'
        return
    import mujoco
    return mujoco.mj_saveModel(model,path,buffer)


def save_image(image,path):
    if owner(path) is None:image.save(path)


def export_final_diagnostics(runtime):
    """After execution only: reports and M2T2 inputs needed for final videos.

    Checkpoints and navigation-map handoffs are never exported.
    """
    for name,value in runtime.artifacts.items():
        path=Path(name)
        if path.name not in {'report.json','chosen.json','input.npz'}:continue
        path.parent.mkdir(parents=True,exist_ok=True)
        if isinstance(value,str):path.write_text(value)
        elif isinstance(value,dict):np.savez_compressed(path,**value)
