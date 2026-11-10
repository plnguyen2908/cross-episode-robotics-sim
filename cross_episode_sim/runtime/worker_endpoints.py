"""Keep stateful navigation history separate between simulator workers."""


def worker_endpoints(gpus, shared, overrides=None):
    if len(set(gpus)) != len(gpus):
        raise ValueError('Duplicate simulator GPU slots')
    overrides = overrides or {}
    unknown = set(overrides)-set(gpus)
    if unknown:
        raise ValueError('Endpoint overrides name unknown GPU slots: '+str(sorted(unknown)))
    result = {gpu: dict(shared, **overrides.get(gpu, {})) for gpu in gpus}
    ring = [endpoints.get('onering') for endpoints in result.values()]
    # Stateless services can share model weights. One-Ring retains the current
    # episode's temporal context; serial RPC handlers do not isolate episodes.
    if len(gpus)>1 and (None in ring or len(set(ring)) != len(ring)):
        raise ValueError('Parallel tasks require a distinct One-Ring endpoint per GPU slot; '
                         'supply --worker-endpoints or use one simulator slot')
    return result
