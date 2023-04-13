"""Custom types declared here"""
from numpy import ndarray
from typing import Tuple, Union, Any, List

# decoder
IsDecoderFrameAvailable = bool
FrameRGB = ndarray
RawMotionVectors = ndarray
FrameType = str

DecodedData = Tuple[
    IsDecoderFrameAvailable,
    FrameRGB,
    RawMotionVectors,
    FrameType
]

# motion vector processor
IsMotionVectorFrameAvailable = bool
MotionVectorFrame = ndarray

# optical flow generator
OpticalFlowFrame = ndarray

# mask generator
IsHumanDetected = bool
SegmentationMask = ndarray
BoundingBoxXY1XY2 = ndarray

MaskOnlyData = Tuple[
    IsHumanDetected,
    SegmentationMask
]
MaskWithMostCenterBoundingBoxData = Tuple[
    IsHumanDetected,
    SegmentationMask,
    BoundingBoxXY1XY2
]

# generals
FrameOfMotionDataType = Union[MotionVectorFrame, OpticalFlowFrame]
ImageType = Union[FrameRGB, FrameOfMotionDataType, SegmentationMask]
ResolutionHW = Union[Tuple[int, int], ndarray]
ResolutionHWC = Union[Tuple[int, int, int], ndarray]
ColorRGB = Union[Tuple[int, int, int], ndarray]
