"""Inference-only reuse of DA3's view-independent computation.

Does not cache cross-view attention or final any-view predictions. Cache keys
include the actual preprocessed pixels, shape and dtype, not frame indices.
Upstream weights and forward methods remain authoritative for new views.
"""
from collections import OrderedDict
import hashlib
import torch


def image_keys(x):
    if x.shape[0] != 1: raise ValueError('Cache supports one video at a time')
    return [hashlib.blake2b(str((tuple(v.shape),v.dtype)).encode()+v.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes(),digest_size=16).digest() for v in x[0]]


def slice_tree(value,index,n):
    if torch.is_tensor(value):
        if value.ndim < 2 or value.shape[1] != n:raise ValueError('Unexpected metric output shape')
        return value[:,index:index+1].clone()
    if isinstance(value,dict):return type(value)({k:slice_tree(v,index,n) for k,v in value.items()})
    raise ValueError('Unexpected non-tensor metric output')


def join_tree(values):
    if torch.is_tensor(values[0]):return torch.cat(values,dim=1)
    return type(values[0])({k:join_tree([v[k] for v in values]) for k in values[0]})


class DA3IndependentCache:
    def __init__(self,predictor,capacity=32):
        self.predictor=predictor;self.capacity=capacity;self.prefix=OrderedDict();self.metric=OrderedDict()
        self.stats=dict(prefix_hits=0,prefix_misses=0,metric_hits=0,metric_misses=0)
        net=predictor.model;backbone=net.da3.backbone.pretrained
        if backbone.alt_start!=13 or backbone.rope_start!=13:raise ValueError('Unqualified DA3 backbone layout')
        self.blocks={id(b):i for i,b in enumerate(backbone.blocks)}
        self.original_prepare=backbone.prepare_tokens_with_masks;self.original_attention=backbone.process_attention
        self.original_metric=net.da3_metric.forward
        self.keys=[]
        def prepare(x,*args,**kwargs):
            if backbone.training or torch.is_grad_enabled():raise RuntimeError('Cache is inference-only')
            self.keys=image_keys(x)
            self.stats['prefix_hits']+=sum(k in self.prefix for k in self.keys)
            self.stats['prefix_misses']+=sum(k not in self.prefix for k in self.keys)
            return self.original_prepare(x,*args,**kwargs)
        def attention(x,block,attn_type='global',pos=None,attn_mask=None):
            i=self.blocks[id(block)]
            if i>=12:return self.original_attention(x,block,attn_type,pos,attn_mask)
            if attn_type!='local' or pos is not None or attn_mask is not None:raise ValueError('Prefix no longer independent')
            missing=[j for j,k in enumerate(self.keys) if k not in self.prefix]
            out=x.clone()
            if missing:out[:,missing]=self.original_attention(x[:,missing],block,'local')
            if i==11:
                for j in missing:self.prefix[self.keys[j]]=out[:,j:j+1].clone()
                out=torch.cat([self.prefix[k] for k in self.keys],dim=1)
                self.trim(self.prefix)
            return out
        def metric(x,*args,**kwargs):
            if args or kwargs:raise ValueError('Metric cache only supports unconditioned inference')
            keys=image_keys(x);missing=list(dict.fromkeys(k for k in keys if k not in self.metric))
            self.stats['metric_hits']+=len(keys)-len(missing);self.stats['metric_misses']+=len(missing)
            if missing:
                indices=[keys.index(k) for k in missing];result=self.original_metric(x[:,indices])
                for j,k in enumerate(missing):self.metric[k]=slice_tree(result,j,len(missing))
            # Cat creates fresh outputs; upstream metric scaling mutates its result.
            out=join_tree([self.metric[k] for k in keys]);self.trim(self.metric);return out
        backbone.prepare_tokens_with_masks=prepare;backbone.process_attention=attention;net.da3_metric.forward=metric

    def trim(self,cache):
        while len(cache)>self.capacity:cache.popitem(last=False)

    def clear(self):
        self.prefix.clear();self.metric.clear()
