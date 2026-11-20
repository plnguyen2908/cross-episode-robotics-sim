"""Qwen facade over the full recipe's existing navigation and atomic handlers."""
import json
import numpy as np
from . import artifacts
from ..code_policy.skill_interface import QwenSkillInterface


DEFAULT_REDOCK=('pick','open_door','close_door','open_drawer','close_drawer','rotate_knob','push_button')


# Appliances whose handle search first stands facing the appliance body when no
# handle on it is in view, and the distance (m) it stands at.
# The fridge was added after fix138: from where the drive to the fridge ended,
# pointing gave two points and neither was the door's vertical bar, no stand
# spot was chosen and the handle was then 'not visible' for the grasp (the run
# had heated and served the food and closed the microwave by then).
FACE_APPLIANCE_FIRST=('microwave','fridge')
FACE_APPLIANCE_DISTANCE=1.0
# Further square-on stands tried when the handle is still not measured after facing the appliance.
# Distance (m) in front of an opened appliance's opening from which an object inside is looked for.
INSIDE_LOOK_DISTANCE=.75
FRONT_VIEW_RETRIES=2

class PipelineRedocking:
    def __init__(self,pipeline,context,fixtures):
        self.pipeline,self.context,self.fixtures=pipeline,context,fixtures
        self.calls=0
        self.next_skill=None

    def __call__(self,*,target,purpose,distance=None,face_first=False):
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
        if purpose=='pick' and target in ('bowl','cup','packaged drink','packaged snack'):
            from ..code_policy.redocking import SensorRedocking
            import numpy as np
            # A handoff saved after a failed call has no navigation report;
            # the redock then looks for the target in its own fresh view.
            report=json.loads(artifacts.read_text(previous/'report.json')) if artifacts.exists(previous/'report.json') else {}
            observed=report.get('selected_object')
            if observed and observed.get('label')==target:
                # Navigation's point is an observed association, not a grasp.
                # Transform its stage frame into this view; refresh from RGB-D
                # before manipulation, without silently selecting another item.
                metadata=json.loads(artifacts.read_text(previous/'slam/mapper.json'))
                old_anchor=np.asarray(metadata.get('global_anchor',np.eye(4)))
                point=session.mapper.inverse@old_anchor@np.r_[observed['position'],1.]
                redocking_class=SensorRedocking
                if target in ('packaged drink','packaged snack'):
                    from ..code_policy.grocery_redocking import GroceryRedocking
                    redocking_class=GroceryRedocking
                session._redocking=redocking_class(p.runtime,session.sensors,session.mapper)
                session._redocking.target_point=point[:3]
                session._redocking.target_label=target
                session._redocking.preserve_target=True
        opened=None
        fixture=target.removeprefix('handle on ')
        # The appliance to stand in front of when its handle or one of its
        # controls is not in view ('Start key at the bottom of the microwave
        # keypad': fix131 stood too near after the close, pointing landed on
        # the robot's own arm, and the key was never pressed).
        face_fixture=fixture if fixture in FACE_APPLIANCE_FIRST else next(
            (name for name in FACE_APPLIANCE_FIRST+('stove',) if name in target.lower() and any(w in target.lower() for w in ('key','button','knob'))),None)
        # 'stove' only for its controls (fix168 call 13: a steep side view of
        # the 7-knob row left the counted knob without usable depth).
        # An object inside the appliance too: with the only pointed spot on the
        # open door refused (fix151: the food was 1.4 m away, out of the
        # wrist view), stand in front of the appliance and look again.
        inside=next((name for name in FACE_APPLIANCE_FIRST if target.lower().endswith(' inside '+name)),None)
        if face_fixture is None and purpose=='pick' and inside is not None:face_fixture=inside
        if purpose=='handle' and getattr(self,'next_skill',None)=='open_door' and 'cabinet' in fixture:
            # Keep the handle navigation chose from its whole-kitchen survey;
            # a second independent pointing from the close-up view picked the
            # small compartment under the stove top (fix57/fix59/fix60).
            report=json.loads(artifacts.read_text(previous/'report.json')) if artifacts.exists(previous/'report.json') else {}
            observed=report.get('selected_object')
            if observed and str(observed.get('label','')).endswith(fixture):
                from ..code_policy.redocking import SensorRedocking
                import numpy as np
                metadata=json.loads(artifacts.read_text(previous/'slam/mapper.json'))
                old_anchor=np.asarray(metadata.get('global_anchor',np.eye(4)))
                point=session.mapper.inverse@old_anchor@np.r_[observed['position'],1.]
                session._redocking=SensorRedocking(p.runtime,session.sensors,session.mapper)
                session._redocking.target_point=point[:3]
                session._redocking.target_label='handle on '+fixture
                session._redocking.preserve_target=True
                print('REDOCK_NAVIGATION_HANDLE',json.dumps(dict(target=fixture,map_point=np.round(point[:3],3).tolist())),flush=True)
        if purpose=='handle' and getattr(self,'next_skill',None) in ('close_door',):
            # A door this robot already opened is closed through its own
            # remembered opened-handle position (observed while opening,
            # reanchored by current SLAM), not by pointing again: live fix58
            # pointed at another, closed cabinet and docked facing away.
            fixtures=self.fixtures()
            observed=(fixtures.aperture_observation(fixture)
                      if fixtures is not None and hasattr(fixtures,'aperture_observation') else None)
            if observed is not None and observed.get('open_handle') is not None:
                from ..code_policy.redocking import SensorRedocking
                import numpy as np
                session._redocking=SensorRedocking(p.runtime,session.sensors,session.mapper)
                session._redocking.target_point_base=np.asarray(observed['open_handle'],float)
                session._redocking.target_label='handle on '+fixture
                session._redocking.preserve_target=True;session._redocking.trust_memory=True
                opened=observed['open_handle']
                print('REDOCK_REMEMBERED_OPEN_HANDLE',json.dumps(dict(target=target,base_point=np.round(opened,3).tolist())),flush=True)
        if purpose=='handle' and getattr(self,'next_skill',None)=='close_drawer':
            # Same for a drawer this robot opened: dock at the handle it
            # remembers holding when the pull ended, not at whichever drawer
            # handle a fresh pointing returns (there are four in this bank).
            fixtures=self.fixtures()
            remembered=(fixtures.drawer_observation(fixture)
                        if fixtures is not None and hasattr(fixtures,'drawer_observation') else None)
            if remembered is not None:
                from ..code_policy.redocking import SensorRedocking
                import numpy as np
                session._redocking=SensorRedocking(p.runtime,session.sensors,session.mapper)
                session._redocking.target_point_base=np.asarray(remembered['open_handle'],float)
                session._redocking.target_label='handle on '+fixture
                session._redocking.preserve_target=True;session._redocking.trust_memory=True
                print('REDOCK_REMEMBERED_DRAWER_HANDLE',json.dumps(dict(target=target,base_point=np.round(remembered['open_handle'],3).tolist())),flush=True)
        original=getattr(p.runtime,'qwen_docking',False)
        try:
            p.runtime.qwen_docking=True
            # Control keys are visual contact targets, not door handles.
            local_purpose='pick' if purpose=='handle' and any(w in target.lower() for w in ('key','button')) else purpose
            if purpose=='pick' and distance is None:
                from .household_components import confined_pick_options
                options=confined_pick_options(target) or {}
                distance=options.get('redock_distance')
            # First look at a face-first appliance: a point that is not on a bar
            # handle, with no bar measured beside it, counts as not seen.
            p.runtime.require_bar_handle=purpose=='handle' and fixture in FACE_APPLIANCE_FIRST
            try:
                # `face_first` (a key that was pointed but could not be located
                # with confidence) asks for a new view: stand in front of the
                # appliance first, as when the target is not seen at all.
                if face_first and face_fixture is not None:raise RuntimeError('Target not visible; a new view was asked for')
                result=session.redock(target,local_purpose,distance)
            except RuntimeError as exc:
                p.runtime.require_bar_handle=False
                if not (purpose in ('handle','pick') and face_fixture is not None and 'not visible' in str(exc)):raise
                # No handle on the appliance in view (fix102: navigation stopped
                # with the microwave half out of frame and only the cabinet
                # handle under it visible). Stand facing the appliance itself,
                # then look for its handle from there.
                print('REDOCK_FACE_APPLIANCE_FIRST',json.dumps(dict(target=target,reason=str(exc)[:80])),flush=True)
                # Count the refused redock as a call of this session: its output
                # folder is numbered by the call count and already exists, so the
                # next redock stopped with FileExistsError (fix114).
                import time
                session.log('redock',dict(target=target,purpose=local_purpose,distance=distance),
                            dict(success=False,reason=str(exc)[:200]),time.monotonic())
                # Stand at the asked distance (ranked docking would close in to 0.65 m).
                p.runtime.qwen_docking=False
                # A look that pointed on the door front without measuring its
                # handle left a stand direction square to the door: use it.
                squared=None
                leaf=(getattr(p.runtime,'open_leaves',None) or {}).get(face_fixture) if purpose=='pick' else None
                if leaf is not None and leaf.get('opening_centre') is not None and session._redocking is not None:
                    # This method imports numpy locally further up on some
                    # paths only, so `np` is a local name here: bind it (fix152
                    # call 18: UnboundLocalError, the opening view never ran).
                    import numpy as np
                    # Look into the opening from in front of it, as close as the
                    # pick stands (fix151 replay: from 1 m in front of the
                    # appliance the food was 1.6 m away and not pointed).
                    session._redocking.front_view_hint=(np.asarray(leaf['opening_centre'],float),np.asarray(leaf['opening_normal'],float))
                    try:squared=session._redocking.front_view_dock(INSIDE_LOOK_DISTANCE,root/'opening_view')
                    except RuntimeError as refused:print('REDOCK_OPENING_VIEW_REFUSED',str(refused)[:120],flush=True)
                    print('REDOCK_OPENING_VIEW',json.dumps(dict(target=target,success=bool(squared and squared.get('success')))),flush=True)
                if squared is None or not squared.get('success'):
                  try:squared=session._redocking.front_view_dock(FACE_APPLIANCE_DISTANCE,root/'front_view') if session._redocking is not None else None
                  except RuntimeError as refused:print('REDOCK_FRONT_VIEW_REFUSED',str(refused)[:120],flush=True)
                try:faced=squared if squared is not None and squared.get('success') else session.redock(face_fixture,'approach',FACE_APPLIANCE_DISTANCE)
                except RuntimeError as unseen:
                    if 'not visible' not in str(unseen):raise
                    # The appliance itself is out of frame: turn in place until it is seen.
                    session.log('redock',dict(target=face_fixture,purpose='approach',distance=FACE_APPLIANCE_DISTANCE),
                                dict(success=False,reason=str(unseen)[:200]),time.monotonic())
                    faced=session._redocking.look_around_for(face_fixture,'approach',FACE_APPLIANCE_DISTANCE,
                        lambda name:root/f'look_around_{name}')
                finally:p.runtime.qwen_docking=True
                print('REDOCK_FACED_APPLIANCE',json.dumps(dict(success=bool(faced.get('success')),reason=str(faced.get('reason'))[:80],
                    final_distance_m=faced.get('final_distance_m'))),flush=True)
                for again in range(FRONT_VIEW_RETRIES+1):
                    try:result=session.redock(target,local_purpose,distance);break
                    except RuntimeError as still:
                        # Still no handle measured from the new stand spot, but
                        # this look left a new square-on direction: stand there
                        # and look again (fix142 call 24: the first square-on
                        # stand had no route from where the drive had jammed
                        # beside the counter, the plain approach then left the
                        # robot 0.9 m to the side of the fridge at 32 deg, and
                        # the opening failed with the handle bars in view).
                        if again==FRONT_VIEW_RETRIES or 'not visible' not in str(still) or session._redocking is None:raise
                        session.log('redock',dict(target=target,purpose=local_purpose,distance=distance),
                                    dict(success=False,reason=str(still)[:200]),time.monotonic())
                        p.runtime.qwen_docking=False
                        try:squared=session._redocking.front_view_dock(FACE_APPLIANCE_DISTANCE,root/f'front_view_{again+1}')
                        except RuntimeError as refused:squared=None;print('REDOCK_FRONT_VIEW_REFUSED',str(refused)[:120],flush=True)
                        finally:p.runtime.qwen_docking=True
                        if squared is None or not squared.get('success'):
                            # The look left no direction at all (replay of the
                            # fix142 state: from the square-on stand in front of
                            # the fridge's hinge side nothing was pointed). Face
                            # the appliance itself again and look from there.
                            p.runtime.qwen_docking=False
                            try:refaced=session.redock(face_fixture,'approach',FACE_APPLIANCE_DISTANCE)
                            except RuntimeError:raise still
                            finally:p.runtime.qwen_docking=True
                            print('REDOCK_FACED_APPLIANCE_AGAIN',json.dumps(dict(success=bool(refaced.get('success')),
                                final_distance_m=refaced.get('final_distance_m'))),flush=True)
                            if not refaced.get('success'):raise still
        finally:
            p.runtime.qwen_docking=original;p.runtime.require_bar_handle=False
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
