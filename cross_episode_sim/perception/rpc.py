"""Local persistent model RPC. No simulator state or executable serialization."""
import json
import os
import time
import numpy as np
import msgpack


def pack(value):
    def default(x):
        if isinstance(x,np.ndarray):
            if x.dtype.hasobject:raise TypeError('Object arrays are not supported')
            x=np.ascontiguousarray(x)
            return {'__array__':True,'dtype':x.dtype.str,'shape':x.shape,'data':x.tobytes()}
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    return msgpack.packb(value,default=default,use_bin_type=True)


def unpack(value):
    def hook(x):
        if x.get('__array__') is True:
            dtype=np.dtype(x['dtype'])
            if dtype.hasobject:raise ValueError('Object arrays forbidden')
            return np.frombuffer(x['data'],dtype=dtype).reshape(x['shape']).copy()
        return x
    return msgpack.unpackb(value,object_hook=hook,raw=False)


CALL_METRICS={}

def call_metrics():
    return {key:dict(value) for key,value in CALL_METRICS.items()}


OPTIONS=[('grpc.max_send_message_length',128*1024*1024),('grpc.max_receive_message_length',128*1024*1024)]


class ModelClient:
    def __init__(self, address, timeout=240, kind=None):
        import grpc
        self.channel=grpc.insecure_channel(address,options=OPTIONS)
        self.request=self.channel.unary_unary('/sensor.Model/Predict',request_serializer=pack,response_deserializer=unpack)
        self.timeout=timeout;self.calls=0;self.wall_seconds=0.;self.kind=kind or address
        self.server_instance=None
    def call(self, **payload):
        start=time.monotonic()
        failed=True
        try:
            # A briefly restarting worker should not abort motion immediately.
            # This waits for a connection within the existing request deadline;
            # it does not blindly replay requests interrupted during inference.
            response=self.request(payload,timeout=self.timeout,wait_for_ready=True)
            instance=response.get('instance')
            if self.kind=='onering' and self.server_instance is not None and instance!=self.server_instance and payload.get('operation')!='reset':
                raise RuntimeError('One-Ring worker restarted; reset the navigation session before continuing')
            self.server_instance=instance
            if not response['ok']:raise RuntimeError(response['error'])
            failed=False
            return response['result']
        finally:
            elapsed=time.monotonic()-start
            self.calls+=1;self.wall_seconds+=elapsed
            stats=CALL_METRICS.setdefault(self.kind,dict(calls=0,wall_seconds=0.,failures=0))
            stats['calls']+=1;stats['wall_seconds']+=elapsed;stats['failures']+=int(failed)
    def close(self):self.channel.close()


def configured_client(kind):
    address=json.loads(os.environ.get('SENSOR_MODEL_ENDPOINTS','{}')).get(kind)
    return ModelClient(address,kind=kind) if address else None


class RemoteSegmenter:
    def __init__(self, client):self.client=client
    def segment(self, image):return self.client.call(image=np.asarray(image))


def start_server(handler,address='127.0.0.1:0'):
    import grpc
    import uuid
    instance=uuid.uuid4().hex
    from concurrent.futures import ThreadPoolExecutor
    # Serialize GPU/model state: One-Ring sessions must be reset explicitly.
    server=grpc.server(ThreadPoolExecutor(max_workers=1),options=OPTIONS)
    def predict(request,context):
        try:
            result=({'ready':True} if request.get('operation')=='health' else handler(request))
            return {'ok':True,'result':result,'instance':instance}
        except Exception as exc:
            return {'ok':False,'error':f'{type(exc).__name__}: {exc}'}
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler('sensor.Model',{
        'Predict':grpc.unary_unary_rpc_method_handler(predict,request_deserializer=unpack,response_serializer=pack)}),))
    port=server.add_insecure_port(address)
    if not port:raise RuntimeError(f'Could not bind {address}')
    server.start();return server,port
