from types import SimpleNamespace
import numpy as np
from cross_episode_sim.mapping.hydra_mapping import prepare_frame,HydraFrameWriter
from cross_episode_sim.perception.visual_targets import Detection


def inputs():
    c=SimpleNamespace(rgb=np.zeros((4,4,3),np.uint8),depth_m=np.ones((4,4)),valid_depth=np.ones((4,4),bool),robot_mask=np.zeros((4,4),bool),K=np.eye(3),T_base_optical=np.eye(4))
    return c,SimpleNamespace(accepted=True,T_map_base=np.eye(4))


def test_hydra_contract_uses_estimated_optical_pose_and_masks_robot():
    c,t=inputs();t.T_map_base[0,3]=2;c.T_base_optical[1,3]=.4;c.robot_mask[0,0]=True
    mask=np.ones((4,4),bool);f=prepare_frame(c,t,[Detection('cup',.9,(0,0,4,4),mask)],1.)
    assert np.allclose(f['T_map_optical'][:3,3],[2,.4,0])
    assert f['depth'][0,0]==0 and f['labels'][0,0]==0
    assert f['labels'][1,1]==1 and f['timestamp_ns']==1000000000
    t.accepted=False;assert prepare_frame(c,t,[],2.) is None


def test_conflicting_predictions_remain_unclassified():
    c,t=inputs();mask=np.ones((4,4),bool)
    d=[Detection('cup',.9,(0,0,4,4),mask),Detection('bowl',.9,(0,0,4,4),mask)]
    assert not prepare_frame(c,t,d,1.)['labels'].any()


def test_writer_does_not_save_rejected_tracking(tmp_path):
    c,t=inputs();w=HydraFrameWriter(tmp_path);t.accepted=False
    assert w.update(c,t,[],1.) is None and not list(tmp_path.glob('frame*'))


def test_semantic_boundary_filter_preserves_geometry_and_removes_depth_bleed():
    from cross_episode_sim.mapping.hydra_mapping import refine_semantics
    depth=np.ones((20,20),np.float32);depth[:,12:]=3.
    labels=np.zeros((20,20),np.int32);labels[3:17,3:14]=1
    before=depth.copy();out=refine_semantics(depth,labels)
    assert out[8,8]==1 and not out[:,12:].any()
    assert not out[3].any() and np.array_equal(depth,before)


def test_receptacle_labels_are_excluded_without_removing_depth():
    from cross_episode_sim.mapping.hydra_mapping import refine_semantics
    c,t=inputs();mask=np.ones((4,4),bool)
    frame=prepare_frame(c,t,[Detection('dining table',.9,(0,0,4,4),mask)],1.)
    assert not frame['labels'].any()
    assert np.array_equal(frame['depth'],c.depth_m)
    # Old saved packets must not recreate furniture object nodes either.
    old=np.full((20,20),3,np.int32);depth=np.ones(old.shape,np.float32)
    assert not refine_semantics(depth,old).any()
    assert np.all(depth==1)
