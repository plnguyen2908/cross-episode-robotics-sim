"""Private evaluator helpers; object ancestry must never feed the policy."""
import numpy as np


def contact_root_lookup(body_parents, object_roots):
    """Map nested collision bodies to their evaluated free-object root."""
    parents=np.asarray(body_parents);roots=set(map(int,object_roots));out=np.full(len(parents),-1,dtype=int)
    for body in range(len(parents)):
        ancestor=body
        while ancestor not in roots and ancestor!=int(parents[ancestor]):
            ancestor=int(parents[ancestor])
        if ancestor in roots:out[body]=ancestor
    return out
