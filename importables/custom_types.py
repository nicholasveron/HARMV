"""Custom types declared here"""
from numpy import ndarray
from typing import Tuple, Union, Any, List
from torch import Tensor

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
ColorRGBXY = Union[Tuple[int, int, int, int, int], ndarray]
ColorXY = Union[Tuple[int, int], ndarray]
ColorInput = Union[ColorRGB, ColorXY, ColorRGBXY]

# interfaces
class NameAdapterInterface:

    def __call__(self, original_name: str) -> Any:
        return self.transform(original_name)

    def transform(self, original_name: str) -> str:
        return original_name

    def inverse_transform(self, original_name: str) -> str:
        return original_name
