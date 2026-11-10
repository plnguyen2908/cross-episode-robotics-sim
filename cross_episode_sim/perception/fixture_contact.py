"""Find a small contact feature in an enlarged, semantically associated fixture crop."""
from .point_masks import PointMaskDetector
from .fixture_pointing import ApplianceAssociatedPointer


class FixtureContactPointer:
    def __init__(self,pointer,fixture):
        self.pointer=pointer;self.fixture=fixture
        self.detector=PointMaskDetector(categories=(fixture,),pointer=ApplianceAssociatedPointer(pointer,fixture))

    def point(self,image,description):
        import numpy as np
        results=[]
        for detection in self.detector.detect(np.asarray(image)):
            x0,y0,x1,y1=map(int,detection.box)
            padding=max(12,round(.2*max(x1-x0,y1-y0)))
            left=max(0,x0-padding);top=max(0,y0-padding)
            right=min(image.width,x1+padding);bottom=min(image.height,y1+padding)
            crop=image.crop((left,top,right,bottom))
            scale=640/max(crop.size)
            enlarged=crop.resize((round(crop.width*scale),round(crop.height*scale)))
            answer=self.pointer.point(enlarged,description)
            for x,y in answer.get('points',[]):
                point=[left+x/enlarged.width*crop.width,top+y/enlarged.height*crop.height]
                if left<=point[0]<right and top<=point[1]<bottom:results.append(point)
        return dict(points=results,raw='contact pointing inside confirmed fixture crop')
