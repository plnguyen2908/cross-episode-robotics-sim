import numpy as np
import pytest
from cross_episode_sim.perception.rpc import ModelClient,start_server,pack,unpack


def test_arrays_roundtrip_without_pickle():
    values={'rgb':np.arange(60,dtype=np.uint8).reshape(4,5,3),'mask':np.eye(3,dtype=bool),'pose':np.eye(4),'text':'cup'}
    result=unpack(pack(values))
    for k,v in values.items():
        if isinstance(v,np.ndarray):assert np.array_equal(result[k],v) and result[k].dtype==v.dtype
        else:assert result[k]==v
    with pytest.raises(TypeError):pack(np.array([object()]))


def test_persistent_worker_multiple_requests_and_errors():
    calls=[]
    def handler(r):
        calls.append(r)
        if r.get('fail'):raise ValueError('bad input')
        return r['image']+1
    server,port=start_server(handler)
    client=ModelClient(f'127.0.0.1:{port}')
    try:
        assert client.call(operation='health')['ready']
        for _ in range(3):assert (client.call(image=np.zeros((2,2)))==1).all()
        with pytest.raises(RuntimeError,match='bad input'):client.call(fail=True)
        assert len(calls)==4
    finally:client.close();server.stop(0).wait()


def test_client_waits_for_temporarily_unavailable_worker():
    import socket,threading,time
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    address=f'127.0.0.1:{port}'
    client=ModelClient(address,timeout=5)
    servers=[]
    def start_later():
        time.sleep(.2)
        servers.append(start_server(lambda r:'ready',address)[0])
    thread=threading.Thread(target=start_later);thread.start()
    try:assert client.call()=='ready'
    finally:
        thread.join();client.close()
        for server in servers:server.stop(0).wait()


def test_onering_restart_requires_session_reset():
    client=ModelClient('127.0.0.1:1',kind='onering')
    responses=iter([{'ok':True,'result':{},'instance':x} for x in ('old','new','new')])
    client.request=lambda *args,**kwargs:next(responses)
    try:
        client.call(operation='reset')
        with pytest.raises(RuntimeError,match='reset the navigation session'):client.call(goal='table')
        client.call(operation='reset')
    finally:client.close()
