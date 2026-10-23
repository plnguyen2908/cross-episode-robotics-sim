"""One slower retry after the atomic runner restores a failed physical trial."""
from .atomic import BoundSkill


class MotionBackoffSkill(BoundSkill):
    def candidates(self, **arguments):
        self.retry=False
        self.failed_placement=None
        yield False
        if self.retry:
            yield True

    def execute_candidate(self, candidate, **arguments):
        c=self.controller
        changes=dict(smooth_action_profile='conservative',smooth_action_speed=.65,
                     smooth_action_contact_speed=.65,smooth_action_nav_speed=.5)
        missing=object()
        saved={k:getattr(c.args,k,missing) for k in changes}
        clearance=getattr(c,'_retry_release_clearance',missing)
        avoid=getattr(c,'_retry_placement_avoid',missing)
        try:
            if candidate:
                for k,v in changes.items():setattr(c.args,k,v)
                if self.name=='place':c._retry_release_clearance=.005
                if self.failed_placement is not None:c._retry_placement_avoid=self.failed_placement.copy()
                c.record(motion_backoff=dict(skill=self.name,attempt=2,
                         reason='prior physical trial failed; checkpoint restored',settings=changes))
            return self.execute(**arguments)
        except RuntimeError as exc:
            reasons=(('Object dropped after clearing pickup support',
                      'Default carry transition lost tucked posture or bilateral grip')
                     if self.name=='navigate' else
                     ('Missing or ambiguous native table support:',
                      'Released object is not supported on destination table',
                      'Released object has not settled',
                      'Basket withdrawal failed:',
                      'Contact servo tracking error:'))
            self.retry=not candidate and any(reason in str(exc) for reason in reasons)
            if self.retry and 'Basket withdrawal failed:' in str(exc):
                self.failed_placement=c.destination_pose.copy()
            raise
        finally:
            for k,v in saved.items():
                if v is missing:c.args.__dict__.pop(k,None)
                else:setattr(c.args,k,v)
            if clearance is missing:c.__dict__.pop('_retry_release_clearance',None)
            else:c._retry_release_clearance=clearance
            if avoid is missing:c.__dict__.pop('_retry_placement_avoid',None)
            else:c._retry_placement_avoid=avoid
