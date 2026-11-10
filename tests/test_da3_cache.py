from types import SimpleNamespace as NS
import torch
from cross_episode_sim.slam.da3_cache import DA3IndependentCache


class Backbone:
    alt_start=13;rope_start=13;training=False
    def __init__(self):self.blocks=[object() for _ in range(40)];self.calls=[]
    def prepare_tokens_with_masks(self,x):return x.clone()
    def process_attention(self,x,block,attn_type='global',pos=None,attn_mask=None):
        i=self.blocks.index(block);self.calls.append((i,x.shape[1]))
        return x+(i+1)*.01 if i<13 else x+x.mean(dim=1,keepdim=True)*.01
    def run(self,x):
        x=self.prepare_tokens_with_masks(x)
        for i,b in enumerate(self.blocks):x=self.process_attention(x,b,'local' if i<13 else 'global')
        return x


class Metric:
    def __init__(self):self.count=0
    def forward(self,x):
        self.count+=x.shape[1]
        return {'depth':x.mean(dim=2),'sky':x.mean(dim=2)*0,'aux':{}}


def fixture():
    b=Backbone();m=Metric();p=NS(model=NS(da3=NS(backbone=NS(pretrained=b)),da3_metric=m));return p,b,m


@torch.inference_mode()
def test_reuse_preserves_output_and_recomputes_cross_view():
    p,b,m=fixture();cache=DA3IndependentCache(p);reference=Backbone()
    for values in [[1.,2.],[2.,3.]]:
        x=torch.tensor(values).reshape(1,2,1,1,1).expand(1,2,3,2,2)
        torch.testing.assert_close(b.run(x),reference.run(x))
        torch.testing.assert_close(m.forward(x)['depth'],x.mean(dim=2))
    assert cache.stats['prefix_misses']==3
    assert m.count==3
    assert sum(n for i,n in b.calls if i==0)==3
    assert sum(n for i,n in b.calls if i==20)==4


@torch.inference_mode()
def test_metric_mutation_does_not_poison_cache():
    p,b,m=fixture();cache=DA3IndependentCache(p);x=torch.ones(1,2,3,2,2)
    first=m.forward(x);first['depth'].mul_(20)
    torch.testing.assert_close(m.forward(x)['depth'],torch.ones(1,2,2,2))
    cache.clear();assert not cache.metric
