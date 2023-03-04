"""Motion vector extractor processes motion vector from video capturer"""

from .video_capturer import VideoCapturerProcessSpawner, FrameData
import cv2
import numba
import numpy
from numpy import ndarray
import time
from multiprocessing import Queue, Process
import os
from collections import deque
import h5py
import queue

MotionVectorData = tuple[bool, ndarray, ndarray]


class MotionVectorExtractor:
    """Motion vector extractor processes motion vector from video capturer"""

    def __init__(self,
                 path: str,
                 bound: int = 32,
                 raw_motion_vectors: bool = False,
                 camera_realtime: bool = False,
                 camera_update_rate: int = 60,
                 camera_buffer_size: int = 0,
                 letterboxed: bool = False,
                 new_shape: int = 640,
                 box: bool = False,
                 color: tuple[int, int, int, int, int] = (114, 114, 114, 128, 128),
                 stride: int = 32,
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

        # letterbox params
        self.__letterboxed: bool = letterboxed
        self.__new_shape: int = new_shape
        self.__color: tuple[int, int, int, int, int] = color
        if raw_motion_vectors:
            self.__color = (*color[:-2], 0, 0)
        self.__stride: int = stride
        self.__box: bool = box

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
        """Static method to rescale raw motion vector data to 0 255 uint8"""
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

    @staticmethod
    @numba.njit(fastmath=True)
    def __letterbox_core(
        shape: tuple[int, int],
        new_shape: int,
        stride: int,
        box: bool,
    ) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
        # Scale ratio (new / old)
        r = max(shape)
        r = new_shape / r

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]  # wh padding
        if not box:
            dw, dh = dw % stride, dh % stride  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        return new_unpad, (top, bottom, left, right)

    @staticmethod
    def letterbox(
            img: ndarray,
            new_shape: int = 640,
            color: tuple[int, ...] = (114, 114, 114),
            stride: int = 32,
            box: bool = False
    ):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]

        new_unpad, (top, bottom, left, right) = MotionVectorExtractor.__letterbox_core(
            shape, new_shape, stride, box
        )

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_NEAREST)  # switch to NEAREST for faster and sharper interp
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img

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

        if self.__letterboxed:
            frame = MotionVectorExtractor.letterbox(
                frame,
                self.__new_shape,
                self.__color[:-2],
                self.__stride,
                self.__box
            )

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
            self.__last_motion_vectors = motion_vectors

        if self.__letterboxed:
            motion_vectors = MotionVectorExtractor.letterbox(
                self.__last_motion_vectors,
                self.__new_shape,
                self.__color[-2:],
                self.__stride,
                self.__box
            )

        return True, frame, motion_vectors

    def stop(self) -> "MotionVectorExtractor":
        """Stop the video capturer thread"""
        self.__video_process.stop()
        return self

    def suicide(self) -> None:
        """Kill by process id, only be used if instance is a process"""
        os.kill(os.getpid(), 9)

    def __del__(self) -> None:
        self.stop()


class MotionVectorExtractorProcessSpawner:
    """Threaded version of MotionVectorExtractor class to fetch realtime (even if drop frame)"""

    def __init__(self,
                 path: str,
                 bound: int = 32,
                 raw_motion_vectors: bool = False,
                 update_rate: int = 60,
                 camera_realtime: bool = False,
                 camera_update_rate: int = 60,
                 camera_buffer_size: int = 0,
                 letterboxed: bool = False,
                 new_shape: int = 640,
                 box: bool = False,
                 color: tuple[int, int, int, int, int] = (114, 114, 114, 128, 128),
                 stride: int = 32,
                 ) -> None:
        self.__queue: Queue = Queue(maxsize=1)
        self.__path: str = path
        self.__bound: int = bound
        self.__raw_motion_vectors: bool = raw_motion_vectors
        self.__update_rate = update_rate
        self.__camera_realtime: bool = camera_realtime
        self.__camera_update_rate: int = camera_update_rate
        self.__camera_buffer_size: int = camera_buffer_size
        self.__letterboxed: bool = letterboxed
        self.__new_shape: int = new_shape
        self.__color: tuple[int, ...] = color
        self.__stride: int = stride
        self.__box: bool = box
        self.__delay: float = 1/update_rate
        self.__run: bool = False
        self.__first_read: bool = False

    def __refresh(self) -> None:
        count_timeout: int = 0
        mvex = MotionVectorExtractor(self.__path,
                                     self.__bound,
                                     self.__raw_motion_vectors,
                                     self.__camera_realtime,
                                     self.__camera_update_rate,
                                     self.__camera_buffer_size,
                                     self.__letterboxed,
                                     self.__new_shape,
                                     self.__box,
                                     self.__color,
                                     self.__stride
                                     )
        data: MotionVectorData = mvex.read()
        self.__queue.put(data)
        timeout_time: int = self.__update_rate*3
        while True:
            start: float = time.perf_counter()
            data = mvex.read()
            try:
                self.__queue.put(data, block=not self.__camera_realtime)
            except queue.Full:
                count_timeout += 1
                if count_timeout >= timeout_time:  # how many frames until it kills itself
                    print("Motion vector extractor timeout, killing process...")
                    mvex.stop()
                    mvex.suicide()
                    return
            else:
                count_timeout = 0
            if not data[0]:
                print("Motion vector extractor source empty, killing process...")
                mvex.stop()
                mvex.suicide()
                return
            end: float = time.perf_counter()
            time.sleep(max(0, self.__delay - (end - start)))

    def read(self) -> MotionVectorData:
        """Pulls motion vector data from motion vector process and returns it"""
        if self.__run and self.__first_read:
            if not self.__data[0]:
                print("Read: Motion vector extractor source empty, killing process...")
                self.stop(False)
                return self.__data
            try:
                self.__data = self.__queue.get(block=not self.__camera_realtime)
            except queue.Empty:
                _ = ""
        self.__first_read = True
        return self.__data

    def start(self) -> "MotionVectorExtractorProcessSpawner":
        """Spawns new process for motion vector extractor"""
        print("Spawning and initializing motion vector extractor process...")
        self.__process: Process = Process(target=self.__refresh, args=(), daemon=False)
        self.__process.start()
        self.__data: MotionVectorData = self.__queue.get()
        self.__run = True
        print("Motion vector extractor process started")
        return self

    def stop(self, manual: bool = True) -> "MotionVectorExtractorProcessSpawner":
        """Kill existing process for motion vector extractor"""
        if self.__run:
            print("Stopping motion vector extractor process...")
            if not self.__camera_realtime and manual:  # manual trigger to prevent auto kill recursion
                while self.__data[0]:
                    self.read()
                os.kill(self.__process.pid, 9)  # type: ignore
            self.__run = False
            self.__first_read = False
            print("Motion vector extractor process stopped")
        return self

    def __del__(self):
        self.stop()


class MotionVectorMocker(MotionVectorExtractor, MotionVectorExtractorProcessSpawner):
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

    def save(self, raw_motion_vector=True, bound: int = -1) -> bool:
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
        self.__h5py_instance.attrs[self.MOTION_VECTORS_ARE_RAW_ATTR] = raw_motion_vector
        self.__h5py_instance.attrs[self.MOTION_VECTORS_BOUND_ATTR] = bound
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

    def start(self) -> "MotionVectorMocker":
        """Ignored functions to eliminate mock error"""
        return self

    def stop(self) -> "MotionVectorMocker":
        """Ignored functions to eliminate mock error"""
        return self

    def suicide(self) -> None:
        """Ignored functions to eliminate mock error"""
        pass
