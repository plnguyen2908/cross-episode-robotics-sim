from .visual_targets import Detection, ObjectDetector, VisualTargetMap
__all__ = ['Detection', 'ObjectDetector', 'VisualTargetMap']

from .point_masks import PointMaskDetector
__all__.append("PointMaskDetector")
