from cross_episode_sim.mapping.multiview_hydra import merge_camera_objects


def test_cross_view_native_positions_and_distinct_objects_are_preserved():
    nav=[dict(id='1',label='bowl',position=[1.,0.,1.])]
    wrist=[dict(id='2',label='bowl',position=[1.02,0.,1.]),dict(id='3',label='bowl',position=[1.,.5,1.]),dict(id='4',label='cup',position=[1.,0.,1.])]
    merged=merge_camera_objects(nav,wrist,'wrist')
    assert [o['id'] for o in merged]==['1','wrist/3','wrist/4']
    assert merged[0]['position']==nav[0]['position']
    assert merged[1]['position']==wrist[1]['position']
    assert len(nav)==1 and wrist[1]['id']=='3'


def test_views_keep_independent_calibration_and_capture_time(monkeypatch,tmp_path):
    from cross_episode_sim.mapping.multiview_hydra import MultiCameraHydraProcess
    import cross_episode_sim.mapping.multiview_hydra as module
    calls=[]
    class Worker:
        def __init__(self,*args,**kwargs):pass
        def update(self,camera,tracking,detections,timestamp):
            calls.append((camera,tracking,detections,timestamp))
            return {'objects':[dict(id='b',label='bowl',position=[1.,.5,1.])]}
    monkeypatch.setattr(module,'NativeHydraProcess',Worker)
    mapper=MultiCameraHydraProcess.__new__(MultiCameraHydraProcess)
    mapper.output=tmp_path;mapper.worker_root=None;mapper.secondary={}
    mapper.update=lambda *args:{'objects':[],'rooms':['native-nav-room']}
    wrist=object();tracking=object();detections={'wrist':['mask']}
    result=mapper.update_many({'navigation':object(),'wrist':wrist},tracking,detections,12.5)
    assert calls==[(wrist,tracking,['mask'],12.5)]
    assert result['rooms']==['native-nav-room']
    assert result['objects'][0]['id']=='wrist/b'
