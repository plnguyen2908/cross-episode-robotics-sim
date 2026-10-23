from types import SimpleNamespace
import pytest
from cross_episode_sim.skills.household_transfer import GraspAlternativeSkill


def test_distinct_grasp_search_survives_controller_rollback():
    c=SimpleNamespace()
    seen=[]
    def prepare(target): c.physically_rejected_annotation_variants=set()
    def execute(target):
        seen.append(set(c.physically_rejected_annotation_variants))
        c.selected_annotation_variant=next(i for i in range(4) if i not in seen[-1])
        raise RuntimeError('lift failed')
    skill=GraspAlternativeSkill(c,'pick',execute,lambda target:True,prepare)
    for candidate in skill.candidates(target='pot'):
        with pytest.raises(RuntimeError,match='lift failed'):
            skill.execute_candidate(candidate,target='pot')
        # The simulator checkpoint restores controller search state too.
        c.physically_rejected_annotation_variants=set()
        del c.selected_annotation_variant
    assert seen==[set(),{0},{0,1}]


def test_no_duplicate_retry_without_selected_grasp():
    c=SimpleNamespace()
    def prepare(target):c.physically_rejected_annotation_variants=set()
    def execute(target):raise RuntimeError('no reachable candidates')
    skill=GraspAlternativeSkill(c,'pick',execute,lambda target:True,prepare)
    attempts=0
    for candidate in skill.candidates(target='pot'):
        attempts+=1
        with pytest.raises(RuntimeError):skill.execute_candidate(candidate,target='pot')
    assert attempts==1
