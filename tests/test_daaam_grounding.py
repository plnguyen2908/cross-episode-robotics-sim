import numpy as np
import pytest
from cross_episode_sim.perception.daaam_grounding import parse_label,masked_views
from cross_episode_sim.perception.visual_targets import Detection,locate_detection
from types import SimpleNamespace

def camera():
    return SimpleNamespace(depth_m=np.ones((20,20))*2,valid_depth=np.ones((20,20),bool),robot_mask=np.zeros((20,20),bool),K=np.array([[10,0,9.5],[0,10,9.5],[0,0,1]]),T_base_optical=np.eye(4))


def test_vlm_schema_rejects_prose_and_unrecognized_labels():
    assert parse_label('```json\n{"category":"other","description":"counter"}\n```')['category']=='other'
    for text in ['It is a table','{"category":"cabinet","description":"x"}','{"category":"cup"}']:
        with pytest.raises(ValueError):parse_label(text)


def test_mask_depth_excludes_background_inside_box():
    c=camera();c.depth_m[:]=4.;mask=np.zeros((20,20),bool);mask[5:10,5:10]=True;c.depth_m[mask]=1.
    result=locate_detection(c,Detection('other',.9,(0,0,20,20),mask),np.eye(4))
    assert result is not None and result[2]==1.


def test_mask_visualization_preserves_context():
    rgb=np.ones((80,80,3),np.uint8)*100;mask=np.zeros((80,80),bool);mask[30:40,30:40]=True
    full,crop=masked_views(rgb,mask)
    assert np.array_equal(np.asarray(full)[0,0],rgb[0,0])
    assert np.asarray(full)[35,35,0]>100 and crop.width<80


def test_wall_sized_mask_is_not_a_navigable_cup():
    c=camera();mask=np.ones((20,20),bool)
    assert locate_detection(c,Detection('cup',.9,(0,0,20,20),mask),np.eye(4)) is None


def test_full_mask_extent_is_checked_before_depth_slice():
    c=camera();c.K[0,0]=c.K[1,1]=100
    c.depth_m[:]=1.;c.depth_m[:5]=4.
    assert locate_detection(c,Detection('cup',.9,(0,0,20,20),np.ones((20,20),bool)),np.eye(4)) is None


def test_mask_reprojection_keeps_neighbors_distinct_and_merges_duplicates():
    from cross_episode_sim.perception.visual_targets import VisualTargetMap
    c=camera();c.frame_id='nav';c.K[0,0]=c.K[1,1]=100
    left=np.zeros((20,20),bool);left[:,1:8]=True
    right=np.zeros((20,20),bool);right[:,12:19]=True
    a=Detection('other',.9,(0,0,20,20),left);b=Detection('other',.8,(0,0,20,20),right)
    tracker=VisualTargetMap();tracking=SimpleNamespace(accepted=True,T_map_base=np.eye(4))
    tracker.update(c,[a,a,b],tracking,0.)
    assert len(tracker.targets)==2
    tracker.update(c,[a,b],tracking,1.)
    assert len(tracker.targets)==2 and all(t.observations==2 for t in tracker.targets)


def test_proposal_gate_rejects_large_background_and_conflicting_labels():
    from cross_episode_sim.perception.daaam_grounding import match_proposals
    boxes=np.array([[0,0,100,100,.9,0],[10,10,20,20,.9,0]],float)
    masks=[np.ones((100,100),bool),np.zeros((100,100),bool)]
    good=Detection('cup',.9,(10,10,20,20))
    b,m,labels=match_proposals(boxes,masks,[good])
    assert len(b)==1 and labels=={0:'cup'} and m[0] is masks[1]
    b,_,_=match_proposals(boxes,masks,[good,Detection('bowl',.8,(10,10,20,20))])
    assert len(b)==0


def test_map_guidance_requires_current_segment_and_agreeing_depth():
    from cross_episode_sim.perception.daaam_grounding import map_guided_proposals
    c=camera();mask=np.ones((20,20),bool)
    target=SimpleNamespace(observations=2,label='cup',position=np.array([0.,0.,2.]))
    boxes=np.array([[0,0,20,20,.8,0.]])
    found=map_guided_proposals(c,[target],np.eye(4),boxes,[mask])
    assert len(found)==1 and found[0].label=='cup'
    c.depth_m[:]=4.
    assert not map_guided_proposals(c,[target],np.eye(4),boxes,[mask])
    c.depth_m[:]=2.;c.robot_mask[:]=True
    assert not map_guided_proposals(c,[target],np.eye(4),boxes,[mask])
    c.robot_mask[:]=False
    assert not map_guided_proposals(c,[target],np.eye(4),boxes,[np.zeros_like(mask)])


def test_unactivated_tracker_candidate_still_requests_inspection():
    from cross_episode_sim.perception.daaam_grounding import DaaamGrounder
    grounder=DaaamGrounder.__new__(DaaamGrounder)
    boxes=np.array([[0,0,20,20,.9,0.]])
    grounder.segmenter=SimpleNamespace(segment=lambda rgb:(boxes,[np.ones((20,20),bool)]))
    grounder.proposer=None
    grounder.trackers={'navigation':SimpleNamespace(update=lambda *args:[])}
    grounder.views={};grounder.max_queries=2
    c=camera();c.rgb=np.zeros((20,20,3),np.uint8);c.frame_id='navigation'
    assert grounder.detect_camera(c,0.)==[]
    assert grounder.needs_confirmation  # A pause, never an invented detection.
    grounder.segmenter=SimpleNamespace(segment=lambda rgb:(np.empty((0,6)),[]))
    assert grounder.detect_camera(c,1.)==[]
    assert not grounder.needs_confirmation


def test_table_map_guidance_requires_current_mask_and_matching_depth():
    from cross_episode_sim.perception.daaam_grounding import map_guided_proposals
    c=camera();mask=np.ones((20,20),bool)
    table=SimpleNamespace(observations=3,label='dining table',position=np.array([0.,0.,2.]))
    boxes=np.array([[0,0,20,20,.8,0.]])
    proposals=map_guided_proposals(c,[table],np.eye(4),boxes,[mask])
    assert len(proposals)==1 and proposals[0].label=='dining table'
    c.depth_m[:]=4.
    assert not map_guided_proposals(c,[table],np.eye(4),boxes,[mask])


def test_floor_mask_cannot_become_table_from_small_elevated_fragment():
    from cross_episode_sim.perception.surfaces import table_surface_mask
    depth=np.full((100,100),2.)
    depth[30:50,30:50]=1.2
    K=np.array([[100.,0,49.5],[0,100.,49.5],[0,0,1.]])
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=2.
    assert not table_surface_mask(depth,np.ones_like(depth,bool),K,T).any()
    # The same measured elevated surface, actually segmented, remains valid.
    mask=depth<2.
    assert table_surface_mask(depth,mask,K,T).sum()==mask.sum()
    # Whole-table detector boxes include legs/background, unlike object masks.
    from cross_episode_sim.perception.visual_targets import table_points
    c=SimpleNamespace(depth_m=depth,valid_depth=np.ones_like(mask),robot_mask=np.zeros_like(mask),K=K,T_base_optical=T)
    assert len(table_points(c,Detection('dining table',.9,(0,0,100,100)),np.eye(4)))>0
    assert len(table_points(c,Detection('dining table',.9,(0,0,100,100),np.ones_like(mask)),np.eye(4)))==0


def test_table_surface_can_include_legs_and_partial_background():
    from cross_episode_sim.perception.surfaces import table_surface_mask
    depth=np.full((100,100),2.);depth[20:80,20:80]=1.2
    K=np.array([[100.,0,49.5],[0,100.,49.5],[0,0,1.]])
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=2.
    mask=table_surface_mask(depth,np.ones_like(depth,bool),K,T)
    assert mask.sum()==3600
