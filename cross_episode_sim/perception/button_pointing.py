"""Locate a small key on an appliance keypad in one camera image.

A key is a few pixels wide in the full image, so the pointed pixel is refined
in enlarged crops. Measured on 14 saved key presses (meal runs fix117-fix138,
wrist camera; private key positions used for scoring only): the pointed pixel
itself was never on the key (3.5-20 cm off); one zoom-in was right 9 times,
returned nothing 3 times (the code then pressed the unrefined spot, 17-20 cm
off) and was wrong twice. Several zoom-ins that have to agree do better. The
mast camera was right 0 times out of 14 with every variant and is not used.
"""
import numpy as np

FIRST_CROPS=(90,120,60)      # half-sizes (px) tried in turn for the first zoom-in
VOTE_CROPS=(40,90,60)        # half-sizes (px) of the further zoom-ins round its answer
AGREE_METRES=.015            # two answers this close name the same key
CROP_SIZE=540


def refine(pointer,image,pixel,target,half,size=CROP_SIZE):
    """The target pointed again in a crop of +/-`half` px round `pixel`, in full-image pixels, or None."""
    x,y=pixel;w,h=image.size
    box=(max(0,int(x)-half),max(0,int(y)-half),min(w,int(x)+half),min(h,int(y)+half))
    if box[2]-box[0]<8 or box[3]-box[1]<8:return None
    points=pointer.point(image.crop(box).resize((size,size)),target).get('points') or []
    if not points:return None
    px,py=points[0];return (box[0]+px*(box[2]-box[0])/size,box[1]+py*(box[3]-box[1])/size)


def vote(spots,agree=AGREE_METRES):
    """Index of the 3D spot most others agree with (first wins ties), or None when no two agree."""
    spots=[None if s is None else np.asarray(s,float) for s in spots]
    best=None;support=0
    for i,a in enumerate(spots):
        if a is None:continue
        count=sum(1 for j,b in enumerate(spots) if j!=i and b is not None and np.linalg.norm(a-b)<=agree)
        if count>support:best,support=i,count
    return best


def locate_key(pointer,image,target,to_point,pixel):
    """(pixel, 3D spot, record) for the key named `target`, starting from a pointed `pixel`; pixel None when unsure.

    `to_point(pixel)` gives the 3D spot under a pixel or raises ValueError.
    """
    def spot(p):
        if p is None:return None
        try:return np.asarray(to_point(p),float)
        except ValueError:return None
    first=None;record=dict(first_crop=None,answers=0)
    for half in FIRST_CROPS:
        first=refine(pointer,image,pixel,target,half)
        if first is not None and spot(first) is not None:record['first_crop']=half;break
        first=None
    if first is None:return None,None,dict(record,reason='no zoom-in answered')
    pixels=[first]+[refine(pointer,image,first,target,half) for half in VOTE_CROPS]
    spots=[spot(p) for p in pixels];record['answers']=sum(s is not None for s in spots)
    chosen=vote(spots)
    if chosen is None:return None,None,dict(record,reason='zoom-ins disagree')
    agreeing=[s for s in spots if s is not None and np.linalg.norm(s-spots[chosen])<=AGREE_METRES]
    record.update(agreeing=len(agreeing),chosen=chosen)
    return pixels[chosen],np.mean(agreeing,axis=0),record


VIEW_EDGE_PIXELS=50


def key_at_view_edge(pixel,size,margin=VIEW_EDGE_PIXELS):
    """True when a located key is within `margin` px of the image border.

    The one wrong vote in the 14 saved presses (fix138, Start: 6 cm off, the
    microwave never turned on) had the keypad half out of the wrist view.
    """
    x,y=pixel;w,h=size
    return bool(x<margin or y<margin or x>w-margin or y>h-margin)
