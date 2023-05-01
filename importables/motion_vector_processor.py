"""Motion Vector Processor processes raw motion vector list into motion vector frames"""

import h5py
import numba
import numpy
from collections import deque
from .utilities import Utilities
from .custom_types import (
    Union,
    ResolutionHW,
    ResolutionHWC,
    RawMotionVectors,
    MotionVectorFrame,
    IsMotionVectorFrameAvailable,
)


class MotionVectorProcessor:
    """Motion Vector Processor processes raw motion vector list into motion vector frames"""

    def __init__(self,
                 input_size: ResolutionHW,
                 target_size: int = 320,
                 bound: int = 32,
                 raw_motion_vectors: bool = False,
                 ) -> None:

        self.__input_size: ResolutionHW = input_size
        self.__target_size = target_size

        # bound param
        self.__raw_motion_vectors: bool = raw_motion_vectors
        self.__bound: int = bound  # bound will be ignored if raw motion vector
        self.__inverse_rgb_2x_bound: float = 255 / (self.__bound * 2)
        self.__half_rgb: int = 128

        # initialize last motion vector
        if self.__raw_motion_vectors:
            self.__last_motion_vectors: MotionVectorFrame = numpy.zeros((*self.__input_size, 2), dtype=numpy.int16)
        else:
            self.__last_motion_vectors: MotionVectorFrame = numpy.ones((*self.__input_size, 2), dtype=numpy.uint8) * 128

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def __process_mv(
        mv_data: RawMotionVectors,
        res: ResolutionHW
    ) -> tuple[IsMotionVectorFrameAvailable, MotionVectorFrame]:

        # check if there is motion vector, if there is, process
        if len(mv_data) == 0:
            return False, numpy.empty((0, 0, 2), dtype=numpy.int16)

        motion_vectors = numpy.zeros((res[0], res[1], 2), dtype=numpy.int16)

        for x in numba.prange(len(mv_data)):
            motion_vector = mv_data[x]
            # if motion vector is forward, then process
            if motion_vector[0] == -1:
                block_size_x: int = motion_vector[1]
                block_size_y: int = motion_vector[2]
                dst_x: int = motion_vector[5]
                dst_y: int = motion_vector[6]
                motion_x: int = motion_vector[7]
                motion_y: int = motion_vector[8]

                str_y: int = int(dst_y-block_size_y*0.5)
                str_x: int = int(dst_x-block_size_x*0.5)
                end_y: int = int(dst_y-block_size_y*0.5+block_size_y)
                end_x: int = int(dst_x-block_size_x*0.5+block_size_x)

                motion_vectors[str_y:end_y, str_x:end_x, 0] = motion_x
                motion_vectors[str_y:end_y, str_x:end_x, 1] = motion_y

        return True, motion_vectors

    def process(self, raw_motion_vector_list: RawMotionVectors) -> MotionVectorFrame:
        """Processes raw motion vector list to a motion vector frame"""

        exist, motion_vectors = self.__process_mv(
            raw_motion_vector_list,
            self.__input_size
        )

        if exist:
            if not self.__raw_motion_vectors:
                motion_vectors = Utilities.bound_motion_frame(
                    motion_vectors,
                    self.__half_rgb,
                    self.__inverse_rgb_2x_bound
                )
            self.__last_motion_vectors = motion_vectors.copy()

        motion_vectors, _, _, _, _, _ = Utilities.letterbox(
            self.__last_motion_vectors.copy(),
            self.__target_size,
            stride=1
        )

        return motion_vectors


class MotionVectorProcessorMocker(MotionVectorProcessor):

    """Writes and reads processed motion vector to and from a file, mocking MotionVectorProcessor behaviour"""

    MOTION_VECTORS_PATH = "motion_vectors"
    MOTION_VECTORS_ARE_RAW_ATTR = "motion_vectors_are_raw"
    MOTION_VECTORS_BOUND_ATTR = "motion_vectors_bound"
    MOTION_VECTORS_HWC_ATTR = "motion_vectors_hwc"
    MOTION_VECTORS_COUNT_ATTR = "motion_vectors_count"

    def __init__(self,
                 h5py_instance: Union[h5py.File, h5py.Group],
                 bound: int = 32,
                 raw_motion_vectors: bool = False,
                 *args,
                 **kwargs
                 ) -> None:
        self.__motion_vectors: deque[MotionVectorFrame] = deque()
        self.__bound: int = bound
        self.__inverse_rgb_2x_bound: float = 255 / (self.__bound * 2)
        self.__half_rgb: int = 128
        self.__raw_motion_vectors: bool = raw_motion_vectors
        self.__h5py_instance: Union[h5py.File, h5py.Group] = h5py_instance
        self.__motion_vectors_type: bool = True
        self.__motion_vectors_hwc: ResolutionHWC = (0, 0, 0)

    def __len__(self):
        return len(self.__motion_vectors)

    def load(self, start_index: int = -1, stop_index: int = -1) -> bool:
        """Loads motion vectors from file to queues"""
        self.flush()

        if not self.MOTION_VECTORS_PATH in self.__h5py_instance:
            return False

        if start_index < 0:
            start_index = 0

        if stop_index < 0:
            stop_index = len(self.__h5py_instance[self.MOTION_VECTORS_PATH])  # type: ignore

        self.__motion_vectors = deque(self.__h5py_instance[self.MOTION_VECTORS_PATH][start_index:stop_index])  # type: ignore

        self.__motion_vectors_type = bool(self.__h5py_instance.attrs[self.MOTION_VECTORS_ARE_RAW_ATTR])
        self.__motion_vectors_hwc = tuple(self.__h5py_instance.attrs[self.MOTION_VECTORS_HWC_ATTR])  # type: ignore

        return True

    def save(self) -> bool:
        """Saves motion vectors in queues to file and flushes the queue"""
        if self.MOTION_VECTORS_PATH in self.__h5py_instance:
            del self.__h5py_instance[self.MOTION_VECTORS_PATH]

        self.__h5py_instance.create_dataset(name=self.MOTION_VECTORS_PATH, data=numpy.array(list(self.__motion_vectors)),
                                            compression="gzip",
                                            )

        if not numpy.allclose(self.__motion_vectors, self.__h5py_instance[self.MOTION_VECTORS_PATH][()]):  # type: ignore
            return False

        self.__h5py_instance.attrs[self.MOTION_VECTORS_ARE_RAW_ATTR] = self.__raw_motion_vectors
        self.__h5py_instance.attrs[self.MOTION_VECTORS_BOUND_ATTR] = -1 if self.__raw_motion_vectors else self.__bound
        self.__h5py_instance.attrs[self.MOTION_VECTORS_HWC_ATTR] = self.__motion_vectors[0].shape
        self.__h5py_instance.attrs[self.MOTION_VECTORS_COUNT_ATTR] = len(self.__motion_vectors)

        self.flush()
        return True

    def append(self, data: MotionVectorFrame) -> None:
        """Appends motion vectors to respective queues"""
        self.__motion_vectors.append(data.copy())

    def process(self, *args, **kwargs) -> MotionVectorFrame:
        """Pops motion vectors from queue simulates MotionVectorProcessor's read"""
        motion_vectors: MotionVectorFrame = numpy.empty((0))
        if len(self.__motion_vectors) <= 0:
            if self.__motion_vectors_type:
                motion_vectors = numpy.zeros(self.__motion_vectors_hwc, dtype=numpy.int16)
            else:
                motion_vectors = numpy.ones(self.__motion_vectors_hwc, dtype=numpy.uint8) * 128
        else:
            motion_vectors: MotionVectorFrame = self.__motion_vectors.popleft()

        if not self.__raw_motion_vectors and self.__motion_vectors_type:
            motion_vectors = Utilities.bound_motion_frame(
                motion_vectors,
                self.__half_rgb,
                self.__inverse_rgb_2x_bound
            )

        return motion_vectors

    def generate_blank(self, size: Union[ResolutionHWC, None] = None) -> MotionVectorFrame:
        
        if not size:
            size = self.__motion_vectors_hwc

        if self.__motion_vectors_type:
            motion_vectors = numpy.zeros(size, dtype=numpy.int16)
        else:
            motion_vectors = numpy.ones(size, dtype=numpy.uint8) * 128

        if not self.__raw_motion_vectors and self.__motion_vectors_type:
            motion_vectors = Utilities.bound_motion_frame(
                motion_vectors,
                self.__half_rgb,
                self.__inverse_rgb_2x_bound
            )

        return motion_vectors

    def flush(self) -> None:
        """Clear all queues"""
        self.__motion_vectors.clear()
        self.__motion_vectors_type = True
