"""Motion vector extractor processes motion vector from video capturer"""

from .video_capturer import VideoCapturerProcessSpawner, FrameData
import cv2
import numba
import numpy
from numpy import ndarray
import time
from multiprocessing import Queue, Process, active_children
import queue
import os
import sys

MotionVectorData = tuple[bool, ndarray, ndarray, ndarray]


@numba.njit(fastmath=True)
def pre_rev_letterbox(
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


def rev_letterbox(
        img: ndarray,
        new_shape: int = 640,
        color: tuple[int, ...] = (114, 114, 114),
        stride: int = 32,
        box: bool = False
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]

    new_unpad, (top, bottom, left, right) = pre_rev_letterbox(
        shape, new_shape, stride, box
    )

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_NEAREST)  # switch to NEAREST for faster and sharper interp
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img


class MotionVectorExtractor:
    """Motion vector extractor processes motion vector from video capturer"""

    def __init__(self,
                 path: str,
                 bound: int,
                 camera_sampling_rate: int,
                 letterboxed: bool = False,
                 new_shape: int = 640,
                 box: bool = False,
                 color: tuple[int, ...] = (114, 114, 114, 127, 127),
                 stride: int = 32,
                 ) -> None:

        # initiate threaded mvextractor VideoCapture
        self.__video_process: VideoCapturerProcessSpawner = VideoCapturerProcessSpawner(path, camera_sampling_rate).start()

        # bound param
        self.__bound: int = bound
        self.__inverse_rgb_bound_2x: float = 255 / (self.__bound * 2)
        self.__inverse_rgb_bound_2x_x_bound: float = self.__inverse_rgb_bound_2x * self.__bound

        # last flows
        self.__last_flows_x: ndarray = numpy.empty((1, 1)).astype(numpy.uint8)
        self.__last_flows_y: ndarray = numpy.empty((1, 1)).astype(numpy.uint8)

        # letterbox params
        self.__letterboxed: bool = letterboxed
        self.__new_shape: int = new_shape
        self.__color: tuple[int, ...] = color
        self.__stride: int = stride
        self.__box: bool = box

    def read(self) -> MotionVectorData:
        """Read and processes next frame"""
        mv_data: FrameData = self.__video_process.read()

        available: bool = False
        frame: ndarray = numpy.empty((1, 1, 1)).astype(numpy.uint8)
        flows_x: ndarray = numpy.empty((1, 1)).astype(numpy.uint8)
        flows_y: ndarray = numpy.empty((1, 1)).astype(numpy.uint8)

        if mv_data[0]:
            available, frame, flows_x, flows_y = self.__process_frame(
                mv_data,
                self.__inverse_rgb_bound_2x_x_bound,
                self.__inverse_rgb_bound_2x
            )

        if available:
            if flows_x.shape == (1, 1):
                if self.__last_flows_x.shape == (1, 1):
                    res: tuple[int, int] = frame.shape[0], frame.shape[1]
                    self.__last_flows_x: ndarray = numpy.ones(res, dtype=numpy.uint8) * 127
                    self.__last_flows_y: ndarray = numpy.ones(res, dtype=numpy.uint8) * 127
                flows_x = self.__last_flows_x
                flows_y = self.__last_flows_y
            if self.__letterboxed:
                stacked_flows = numpy.dstack(
                    (flows_x, flows_y)
                )
                stacked_flows = rev_letterbox(
                    stacked_flows,
                    self.__new_shape,
                    self.__color[-2:],
                    self.__stride,
                    self.__box
                )
                frame = rev_letterbox(
                    frame,
                    self.__new_shape,
                    self.__color[:-2],
                    self.__stride,
                    self.__box
                )
                flows_x = stacked_flows[:, :, 0]
                flows_y = stacked_flows[:, :, 1]

        return available, frame, flows_x, flows_y

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def __process_frame(
        mv_data: FrameData,
        inverse_rgb_bound_2x_x_bound: float,
        inverse_rgb_bound_2x: float
    ) -> MotionVectorData:
        flows_x: ndarray = numpy.empty((1, 1)).astype(numpy.uint8)
        flows_y: ndarray = numpy.empty((1, 1)).astype(numpy.uint8)

        frame: ndarray = mv_data[1]
        res: tuple[int, int] = frame.shape[0], frame.shape[1]

        # check if there is motion vector, if there is, process
        if len(mv_data[2]) != 0:
            flows_x: ndarray = numpy.zeros(res, dtype=numpy.int16)
            flows_y: ndarray = numpy.zeros(res, dtype=numpy.int16)

            for x in numba.prange(len(mv_data[2])):
                motion_vector = mv_data[2][x]
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

                    flows_x[str_y:end_y, str_x:end_x] = motion_x
                    flows_y[str_y:end_y, str_x:end_x] = motion_y

            # rescale motion vector with bound to 0 255
            # bound is rephrased
            # translate to 255 first from 2*bound
            # then if < 0 which is lower than bound and > 1 which is higher than bound

            flows_x: ndarray = inverse_rgb_bound_2x * flows_x + inverse_rgb_bound_2x_x_bound
            flows_y: ndarray = inverse_rgb_bound_2x * flows_y + inverse_rgb_bound_2x_x_bound

            for x in numba.prange(res[0]):
                for y in numba.prange(res[1]):
                    if flows_x[x][y] < 0:
                        flows_x[x][y] = 0
                    if flows_y[x][y] < 0:
                        flows_y[x][y] = 0
                    if flows_x[x][y] > 255:
                        flows_x[x][y] = 255
                    if flows_y[x][y] > 255:
                        flows_y[x][y] = 255

            flows_x: ndarray = flows_x.astype(numpy.uint8)
            flows_y: ndarray = flows_y.astype(numpy.uint8)

        return True, frame, flows_x, flows_y

    def stop(self) -> "MotionVectorExtractor":
        """Stop the video capturer thread"""
        self.__video_process.stop()
        return self

    def suicide(self) -> None:
        os.kill(os.getpid(), 9)

    def __del__(self) -> None:
        self.stop()


class MotionVectorExtractorProcessSpawner:
    """Threaded version of MotionVectorExtractor class to fetch realtime (even if drop frame)"""

    def __init__(self,
                 path: str,
                 bound: int,
                 sampling_rate: int,
                 camera_sampling_rate: int,
                 letterboxed: bool = False,
                 new_shape: int = 640,
                 color: tuple[int, ...] = (114, 114, 114, 127, 127),
                 stride: int = 32,
                 box: bool = False
                 ) -> None:
        self.queue: Queue = Queue(maxsize=1)
        self.path: str = path
        self.bound: int = bound
        self.sampling_rate = sampling_rate
        self.camera_sampling_rate: int = camera_sampling_rate
        self.letterboxed: bool = letterboxed
        self.new_shape: int = new_shape
        self.color: tuple[int, ...] = color
        self.stride: int = stride
        self.box: bool = box
        self.delay: float = 1/sampling_rate
        self.run: bool = False

    def __refresh(self) -> None:
        count_timeout: int = 0
        mvex = MotionVectorExtractor(self.path,
                                     self.bound,
                                     self.camera_sampling_rate,
                                     self.letterboxed,
                                     self.new_shape,
                                     self.box,
                                     self.color,
                                     self.stride
                                     )
        self.data: MotionVectorData = mvex.read()
        self.queue.put(self.data)
        while True:
            start: float = time.perf_counter()
            self.data = mvex.read()
            if not self.data[0]:
                print("Motion vector extractor source empty, killing process...")
                mvex.stop()
                mvex.suicide()
                return
            try:
                self.queue.put_nowait(self.data)
            except queue.Full:
                count_timeout += 1
                if count_timeout >= self.sampling_rate*3:  # how many frames until it kills itself
                    print("Motion vector extractor timeout, killing process...")
                    mvex.stop()
                    mvex.suicide()
                    return
            else:
                count_timeout = 0
            end: float = time.perf_counter()
            time.sleep(max(0, self.delay - (end - start)))

    def read(self) -> MotionVectorData:
        if self.run:
            try:
                self.data: MotionVectorData = self.queue.get_nowait()
            except queue.Empty:
                _ = ""
        return self.data

    def start(self) -> "MotionVectorExtractorProcessSpawner":
        """Spawns new process for motion vector extractor"""
        print("Spawning and initializing motion vector extractor process...")
        self.process: Process = Process(target=self.__refresh, args=(), daemon=False)
        self.process.start()
        self.data: MotionVectorData = self.queue.get()
        self.run = True
        print("Motion vector extractor process started")
        return self

    def stop(self) -> "MotionVectorExtractorProcessSpawner":
        """Kill existing process for motion vector extractor"""
        if self.run:
            print("Stopping motion vector extractor process...")
            self.run = False
            print("Motion vector extractor process stopped")
        return self

    def __del__(self):
        self.stop()
