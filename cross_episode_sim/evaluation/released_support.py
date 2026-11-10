"""Private semantic scoring of a completed release, never execution feedback."""


def released_support_success(samples):
    """The same initially held object must remain supported and ungrasped.

    Velocity estimates are diagnostic only: contact solver jitter must not
    override sustained physical support. Callers collect a full scoring window
    after the observation-based release/settling procedure.
    """
    if not samples:
        return False
    eligible = set(samples[0]['initially_held'])
    for sample in samples:
        eligible &= (set(sample['initially_held']) & set(sample['supported'])
                     - set(sample['gripper_contact']))
    return bool(eligible)
