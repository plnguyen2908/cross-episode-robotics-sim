from types import SimpleNamespace as NS
import numpy as np
from cross_episode_sim.perception.point_masks import PointMaskDetector


def test_points_use_masks_and_preserve_rgb_bgr_contract():
    rgb=np.zeros((20,20,3),np.uint8);rgb[:,:,0]=201
    mask=np.zeros((20,20),bool);mask[3:12,4:13]=True
    seen=[]
    def point(image,label):
        assert np.asarray(image)[0,0].tolist()==[201,0,0]
        return {'points':[(8,8),(8,8),(-1,2),(20,2),(float('nan'),2)]}
    def segment(image):
        seen.append(image);assert image[0,0].tolist()==[0,0,201]
        return [],[mask]
    detector=PointMaskDetector(('bowl',),pointer=NS(point=point),segmenter=NS(segment=segment))
    result=detector.detect(rgb)
    assert len(result)==1 and len(seen)==1
    assert result[0].label=='bowl' and result[0].box==(4.,3.,13.,12.)
    assert np.array_equal(result[0].mask,mask)
    detector.detect(rgb,masks=[mask]);assert len(seen)==1


def test_no_point_or_mask_does_not_fabricate_detection():
    def unexpected(image):raise AssertionError('Unnecessary segmentation')
    detector=PointMaskDetector(('cup',),pointer=NS(point=lambda *_:{'points':[]}),segmenter=NS(segment=unexpected))
    assert detector.detect(np.zeros((20,20,3),np.uint8))==[]
    detector.pointer=NS(point=lambda *_:{'points':[(8,8)]})
    assert detector.detect(np.zeros((20,20,3),np.uint8),masks=[])==[]


def test_free_support_prompt_keeps_receptacle_label():
    queries=[]
    def point(image,target):
        queries.append(target)
        return {'points':[(8,8)]}
    detector=PointMaskDetector(('dining table',),pointer=NS(point=point),
                              segmenter=object(),free_support=True)
    result=detector.detect(np.zeros((20,20,3),np.uint8),
                           masks=[np.ones((20,20),bool)])
    assert 'empty, unobstructed spot' in queries[0]
    assert 'dining table' in queries[0]
    assert result[0].label=='dining table'
    assert result[0].point==(8,8)


def test_conflicting_labels_for_same_mask_are_rejected():
    detector=PointMaskDetector(('cup','bowl'),pointer=NS(point=lambda *_:{'points':[(8,8)]}),segmenter=object())
    assert detector.detect(np.zeros((20,20,3),np.uint8),masks=[np.ones((20,20),bool)])==[]


def test_tracking_preserves_point_selected_mask_with_same_bounding_box():
    from cross_episode_sim.perception.daaam_grounding import match_proposals
    from cross_episode_sim.perception.visual_targets import Detection
    a=np.eye(10,dtype=bool);b=np.ones((10,10),bool)
    boxes=np.array([[0,0,10,10,.9,0],[0,0,10,10,.8,0]])
    _,masks,labels=match_proposals(boxes,[a,b],[Detection('bowl',1.,(0,0,10,10),b)])
    assert len(masks)==1 and np.array_equal(masks[0],b) and labels=={0:'bowl'}
