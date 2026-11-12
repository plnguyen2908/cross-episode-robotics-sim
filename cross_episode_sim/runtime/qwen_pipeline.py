"""Qwen facade over the full recipe's existing navigation and atomic handlers."""
import json
from . import artifacts
from ..code_policy.skill_interface import QwenSkillInterface


DEFAULT_REDOCK=('pick','open_door','close_door','open_drawer','close_drawer','rotate_knob','push_button')


class PipelineRedocking:
    def __init__(self,pipeline,context,fixtures):
        self.pipeline,self.context,self.fixtures=pipeline,context,fixtures
        self.calls=0

    def __call__(self,*,target,purpose,distance=None):
        from .household_components import live_session
        from ..mapping.continuity import load_memory,save_memory
        p=self.pipeline;ctx=self.context
        if purpose=='pick' and ctx.get('held') is not None:
            return dict(success=False,reason='placement_pending_retained_grasp')
        previous=ctx.get('continuation')
        if previous is None:raise RuntimeError('No live navigation handoff for Qwen docking')
        self.calls+=1;root=p.output/f'qwen_redock_{self.calls:03d}'
        print('QWEN_PIPELINE_REDOCK',purpose,target,str(root),flush=True)
        session=live_session(p.runtime,root,previous,self.fixtures(),target,
                             held=ctx.get('held') if purpose=='place' else None)
        if purpose=='pick' and target in ('bowl','cup'):
            from ..code_policy.redocking import SensorRedocking
            import numpy as np
            report=json.loads(artifacts.read_text(previous/'report.json'))
            observed=report.get('selected_object')
            if observed and observed.get('label')==target:
                # Navigation's point is an observed association, not a grasp.
                # Transform its stage frame into this view; refresh from RGB-D
                # before manipulation, without silently selecting another item.
                metadata=json.loads(artifacts.read_text(previous/'slam/mapper.json'))
                old_anchor=np.asarray(metadata.get('global_anchor',np.eye(4)))
                point=session.mapper.inverse@old_anchor@np.r_[observed['position'],1.]
                session._redocking=SensorRedocking(p.runtime,session.sensors,session.mapper)
                session._redocking.target_point=point[:3]
                session._redocking.target_label=target
                session._redocking.preserve_target=True
        original=getattr(p.runtime,'qwen_docking',False)
        try:
            p.runtime.qwen_docking=True
            # Control keys are visual contact targets, not door handles.
            local_purpose='pick' if purpose=='handle' and any(w in target.lower() for w in ('key','button')) else purpose
            if purpose=='pick' and distance is None:
                from .household_components import confined_pick_options
                options=confined_pick_options(target) or {}
                distance=options.get('redock_distance')
            result=session.redock(target,local_purpose,distance)
        finally:
            p.runtime.qwen_docking=original
            # Keep actual post-motion state and map even when local planning
            # fails. Never reuse a pre-redock frame for subsequent manipulation.
            handoff=session.checkpoint(root/'handoff')
            save_memory(handoff,load_memory(previous))
            ctx['continuation']=handoff
            redocking=session._redocking
            if redocking is not None and redocking.target_point is not None and getattr(redocking,'target_confirmed',False):
                artifacts.write_text(handoff/'report.json',json.dumps(dict(
                    initialization_valid=True,selected_object=dict(label=target,
                    position=redocking.target_point.tolist(),frame='estimated_slam_map'),
                    target_provenance='fresh sensor observation inside Qwen redock')))
            ctx['dock']=handoff
        if purpose=='place':
            ctx['qwen_prepared_session']=session
            ctx['carry']=handoff
        return result


def full_task_interface(skills,pipeline,context,fixtures=lambda:None):
    """Retain supplied cross-room/explore handlers; enable local manipulation docking.

    Placement handlers own loaded Qwen docking and interior refinement already;
    do not prepend a duplicate empty-hand dock. Explicit place redock is routed
    to the same HouseholdSession used by component tests.
    """
    return QwenSkillInterface(skills,PipelineRedocking(pipeline,context,fixtures),
                              default_redock=DEFAULT_REDOCK)
