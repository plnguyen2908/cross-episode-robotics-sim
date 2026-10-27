"""Precision-preserving target deltas for native-controller replay qualification.

These NumPy longdouble recordings are a replay diagnostic format, not a claim
of lossless float32 VLA export. Fail explicitly if a source cannot round-trip.
"""
import numpy as np


def encode_target_deltas(initial, targets):
    targets=np.asarray(targets,dtype=np.float64)
    reference=np.asarray(initial,dtype=np.longdouble).copy()
    if targets.ndim!=2 or reference.shape!=targets.shape[1:]:
        raise ValueError('Expected initial (D,) and targets (T,D)')
    if not np.isfinite(targets).all() or not np.isfinite(reference).all():
        raise ValueError('Nonfinite target')
    result=np.empty(targets.shape,dtype=np.longdouble)
    for i,target in enumerate(targets):
        # Account for representational residue in the decoder's previous
        # target, instead of subtracting an ideal reference it does not hold.
        result[i]=target.astype(np.longdouble)-reference
        reference+=result[i]
        if not np.array_equal(reference.astype(np.float64),target):
            raise ValueError(f'Delta precision cannot reconstruct target at step {i}')
    return result
