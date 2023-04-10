"""Motion vector extractor processes motion vector from video capturer"""

import os
import cv2
import h5py
import time
import numba
import numpy
import queue
from numpy import ndarray
from collections import deque
from multiprocessing import Queue, Process
from .video_capturer import VideoCapturerProcessSpawner, FrameData

MotionVectorData = tuple[bool, ndarray, ndarray]


class MotionVectorExtractor:
    """Motion vector extractor processes motion vector from video capturer"""

    def __init__(self,
                 path: str,
                 bound: int = 32,
                 raw_motion_vectors: bool = False,
                 camera_realtime: bool = False,
                 camera_update_rate: int = 60,
                 camera_buffer_size: int = 0
                 ) -> None:

        # initiate threaded mvextractor VideoCapture
        self.__video_process: VideoCapturerProcessSpawner = VideoCapturerProcessSpawner(
            path,
            camera_realtime,
            camera_update_rate,
            camera_buffer_size
        ).start()

        # bound param
        self.__raw_motion_vectors: bool = raw_motion_vectors
        self.__bound: int = bound  # bound will be ignored if raw motion vector
        self.__inverse_rgb_2x_bound: float = 255 / (self.__bound * 2)
        self.__half_rgb: int = 128

        # shared memory
        self.__initialized: bool = False

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def __process_mv(
        mv_data: ndarray,
        res: tuple[int, int]
    ) -> tuple[bool, ndarray]:

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

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def rescale_mv(
            motion_vectors: ndarray,
            half_rgb: int,
            inverse_rgb_2x_bound: float
    ) -> ndarray:
        """THIS FUNCTION CHANGES THE ORIGINAL INPUT 
        Static method to rescale raw motion vector data to 0 255 uint8"""
        # rescale motion vector with bound to 0 255
        # bound is refactored to precalculate most of the constant
        # translate to 255 first from 2*bound
        # then if < 0 which is lower than bound and > 1 which is higher than bound
        # from int16 to uint8
        for x in numba.prange(motion_vectors.shape[0]):
            for y in numba.prange(motion_vectors.shape[1]):
                motion_vector_x = (inverse_rgb_2x_bound * motion_vectors[x, y, 0]) + half_rgb
                motion_vector_y = (inverse_rgb_2x_bound * motion_vectors[x, y, 1]) + half_rgb

                motion_vectors[x, y, 0] = max(min(motion_vector_x, 255), 0)
                motion_vectors[x, y, 1] = max(min(motion_vector_y, 255), 0)

        motion_vectors = motion_vectors.astype(numpy.uint8)

        return motion_vectors

    def read(self) -> MotionVectorData:
        """Read and processes next frame"""

        fr_data: FrameData = self.__video_process.read()

        available: bool = fr_data[0]

        if not available:
            return False, numpy.empty((0)), numpy.empty((0))

        frame = fr_data[1]

        if not self.__initialized:
            self.__res: tuple[int, int] = frame.shape[0], frame.shape[1]
            if self.__raw_motion_vectors:
                self.__last_motion_vectors: ndarray = numpy.zeros((*self.__res, 2), dtype=numpy.int16)
            else:
                self.__last_motion_vectors: ndarray = numpy.ones((*self.__res, 2), dtype=numpy.uint8) * 128
            self.__initialized: bool = True

        exist, motion_vectors = self.__process_mv(
            fr_data[2],
            self.__res
        )

        if exist:
            if not self.__raw_motion_vectors:
                motion_vectors = self.rescale_mv(
                    motion_vectors,
                    self.__half_rgb,
                    self.__inverse_rgb_2x_bound
                )
            self.__last_motion_vectors = motion_vectors.copy()


        return True, frame, self.__last_motion_vectors.copy()

    def stop(self) -> "MotionVectorExtractor":
        """Stop the video capturer thread"""
        self.__video_process.stop()
        return self

    def suicide(self) -> None:
        """Kill by process id, only be used if instance is a process"""
        os.kill(os.getpid(), 9)

    def __del__(self) -> None:
        self.stop()

class MotionVectorMocker(MotionVectorExtractor):
    """Writes and reads processed motion vector to and from a file, mocking MotionVectorExtractor behaviour"""

    FRAME_PATH = "frames"
    FRAME_HWC_ATTR = "frame_hwc"
    FRAME_COUNT_ATTR = "frame_count"
    MOTION_VECTORS_PATH = "motion_vectors"
    MOTION_VECTORS_ARE_RAW_ATTR = "motion_vectors_are_raw"
    MOTION_VECTORS_BOUND_ATTR = "motion_vectors_bound"
    MOTION_VECTORS_HWC_ATTR = "motion_vectors_hwc"
    MOTION_VECTORS_COUNT_ATTR = "motion_vectors_count"

    def __init__(self,
                 h5py_instance: h5py.File,
                 bound: int = 32,
                 raw_motion_vector: bool = False,
                 *args,
                 **kwargs
                 ) -> None:
        self.__frame: deque[ndarray] = deque()
        self.__motion_vectors: deque[ndarray] = deque()
        self.__bound: int = bound
        self.__inverse_rgb_2x_bound: float = 255 / (self.__bound * 2)
        self.__half_rgb: int = 128
        self.__raw_motion_vector: bool = raw_motion_vector
        self.__h5py_instance: h5py.File = h5py_instance
        self.__motion_vectors_type = True

    def load(self) -> bool:
        """Loads frames and motion vectors from file to queues"""
        self.flush()

        if not all([x in self.__h5py_instance for x in [self.FRAME_PATH, self.MOTION_VECTORS_PATH]]):
            return False

        for frame in self.__h5py_instance[self.FRAME_PATH][()]:  # type: ignore
            self.__frame.append(frame)
        for motion_vectors in self.__h5py_instance[self.MOTION_VECTORS_PATH][()]:  # type: ignore
            self.__motion_vectors.append(motion_vectors)

        if len(self.__frame) != len(self.__motion_vectors):
            self.flush()
            return False

        self.__motion_vectors_type = bool(self.__h5py_instance.attrs[self.MOTION_VECTORS_ARE_RAW_ATTR])

        return True

    def save(self, raw_motion_vectors=True, bound: int = -1) -> bool:
        """Saves frames and motion vectors in queues to file and flushes the queue"""
        if self.FRAME_PATH in self.__h5py_instance:
            del self.__h5py_instance[self.FRAME_PATH]
        if self.MOTION_VECTORS_PATH in self.__h5py_instance:
            del self.__h5py_instance[self.MOTION_VECTORS_PATH]

        self.__h5py_instance.create_dataset(name=self.FRAME_PATH, data=numpy.array(list(self.__frame)),
                                            compression="gzip",
                                            )
        self.__h5py_instance.create_dataset(name=self.MOTION_VECTORS_PATH, data=numpy.array(list(self.__motion_vectors)),
                                            compression="gzip",
                                            )

        if not all(
            numpy.allclose(a, self.__h5py_instance[b][()]) for a, b in [  # type: ignore
                (self.__frame, self.FRAME_PATH),
                (self.__motion_vectors, self.MOTION_VECTORS_PATH)
            ]
        ):
            return False

        self.__h5py_instance.attrs[self.FRAME_HWC_ATTR] = self.__frame[0].shape
        self.__h5py_instance.attrs[self.FRAME_COUNT_ATTR] = len(self.__frame)
        self.__h5py_instance.attrs[self.MOTION_VECTORS_ARE_RAW_ATTR] = raw_motion_vectors
        self.__h5py_instance.attrs[self.MOTION_VECTORS_BOUND_ATTR] = -1 if raw_motion_vectors else bound
        self.__h5py_instance.attrs[self.MOTION_VECTORS_HWC_ATTR] = self.__motion_vectors[0].shape
        self.__h5py_instance.attrs[self.MOTION_VECTORS_COUNT_ATTR] = len(self.__motion_vectors)

        self.flush()
        return True

    def append(self, data: MotionVectorData) -> None:
        """Appends frame and motion vectors to respective queues"""
        if data[0]:
            self.__frame.append(data[1].copy())
            self.__motion_vectors.append(data[2].copy())

    def read(self) -> MotionVectorData:
        """Pops frame and motion vectors from queue simulates MotionVectorExtractor's read"""
        if len(self.__frame) <= 0:
            return False, numpy.empty((0)), numpy.empty((0))
        frame: ndarray = self.__frame.popleft()
        motion_vectors: ndarray = self.__motion_vectors.popleft()
        if not self.__raw_motion_vector and self.__motion_vectors_type:
            motion_vectors = self.rescale_mv(
                motion_vectors,
                self.__half_rgb,
                self.__inverse_rgb_2x_bound
            )
        return True, frame, motion_vectors

    def flush(self) -> None:
        """Clear all queues"""
        self.__frame.clear()
        self.__motion_vectors.clear()
        self.__motion_vectors_type = True

    def stop(self) -> "MotionVectorMocker":
        """Ignored functions to eliminate mock error"""
        return self

    def suicide(self) -> None:
        """Ignored functions to eliminate mock error"""
        pass
