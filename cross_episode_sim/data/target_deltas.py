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


def encode_target_velocities(initial, targets, dt):
    """Lossless rational target-path rates for replay qualification.

    ASCII numerator/denominator rates preserve division by the binary float
    timestep exactly. Training export to numeric float32 is a separate step.
    """
    from fractions import Fraction
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    targets=np.asarray(targets,dtype=np.float64)
    initial=np.asarray(initial,dtype=np.float64)
    if targets.ndim!=2 or initial.shape!=targets.shape[1:]:
        raise ValueError('Expected initial (D,) and targets (T,D)')
    if not np.isfinite(targets).all() or not np.isfinite(initial).all():
        raise ValueError('Nonfinite target')
    timestep=Fraction.from_float(float(dt))
    reference=[Fraction.from_float(float(x)) for x in initial]
    result=np.empty(targets.shape,dtype='S256')
    for i,target in enumerate(targets):
        for j,value in enumerate(target):
            goal=Fraction.from_float(float(value))
            rate=(goal-reference[j])/timestep
            reference[j]+=rate*timestep
            if float(reference[j])!=value:
                raise ValueError(f'Velocity precision cannot reconstruct target at step {i}')
            encoded=str(rate).encode('ascii')
            if len(encoded)>256:raise ValueError('Rate exceeds lossless storage width')
            result[i,j]=encoded
    return result
