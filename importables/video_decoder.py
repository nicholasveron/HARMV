"""Video decoder that decodes video Frames and bounded Motion Vectors images"""

import cv2
import numba
import numpy
from numpy import ndarray
import time
from threading import Thread
from mvextractor.videocap import VideoCap  # pylint: disable=no-name-in-module

DecodedVideoData = tuple[bool, ndarray, ndarray, ndarray]
MotionVectorData = tuple[bool, ndarray, ndarray, str, float]


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


class ThreadedVideoCapture:
    """An improved VideoCap class to fetch realtime (even if drop frame)"""

    def __init__(self, path: str, target_fps: int) -> None:
        print("Initializing video capturer thread ...")
        self.stream = VideoCap()
        self.stream.open(path)
        if not self.stream.open(path):
            assert False, "ERROR WHILE LOADING " + path
        self.mv_data: MotionVectorData = self.stream.read()
        self.stopped = False
        self.delay = 1/target_fps
        print("Video capturer thread initialized")

    def start(self):
        """Start new thread for video capturer"""
        print("Starting video capturer thread ...")
        Thread(target=self.get, args=()).start()
        print("Video capturer thread started")
        return self

    def get(self) -> None:
        """Read next frame into memory"""
        while not self.stopped:
            time.sleep(self.delay)
            if not self.mv_data[0]:
                self.stop()
            else:
                self.mv_data = self.stream.read()

    def read(self) -> MotionVectorData:
        """Returns latest frame"""
        return self.mv_data

    def stop(self) -> None:
        """Release memory when not used anymore"""
        print("Stopping video capturer thread ...")
        self.stopped = True
        self.stream.release()
        print("Video capturer thread stopped")

    def __del__(self) -> None:
        """Release memory when not used anymore"""
        print("Stopping video capturer thread ...")
        self.stopped = True
        self.stream.release()
        print("Video capturer thread stopped")


class VideoDecoder:
    """Video decoder that decodes video Frames and bounded Motion Vectors images"""

    def __init__(self,
                 path: str,
                 bound: int,
                 target_fps: int,
                 letterboxed: bool = False,
                 new_shape: int = 640,
                 color: tuple[int, ...] = (114, 114, 114, 127, 127),
                 stride: int = 32,
                 box: bool = False
                 ) -> None:

        print("Initializing video decoder ...")

        # initiate threaded mvextractor VideoCapture
        self.__video_thread: ThreadedVideoCapture = ThreadedVideoCapture(path, target_fps).start()

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

        print("Video decoder initialized")

    def read(self) -> DecodedVideoData:
        """Reads next frame into memory"""
        mv_data: MotionVectorData = self.__video_thread.mv_data

        available: bool = False
        frame: ndarray = numpy.empty((1, 1, 1)).astype(numpy.uint8)
        flows_x: ndarray = numpy.empty((1, 1)).astype(numpy.uint8)
        flows_y: ndarray = numpy.empty((1, 1)).astype(numpy.uint8)

        if mv_data[0]:
            available, frame, flows_x, flows_y = self.__process_frame(
                mv_data,
                self.__bound,
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
        mv_data: MotionVectorData,
        bound: int,
        inverse_rgb_bound_2x_x_bound: float,
        inverse_rgb_bound_2x: float
    ) -> DecodedVideoData:
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

    def stop(self) -> None:
        """Stop the video capturer thread"""
        self.__video_thread.stop()

    def __del__(self) -> None:
        self.__video_thread.stop()


class ThreadedVideoDecoder:
    """Threaded version of VideoDecoder class to fetch realtime (even if drop frame)"""

    def __init__(self,
                 path: str,
                 bound: int,
                 target_fps: int,
                 letterboxed: bool = False,
                 new_shape: int = 640,
                 color: tuple[int, ...] = (114, 114, 114, 127, 127),
                 stride: int = 32,
                 box: bool = False
                 ) -> None:
        print("Initializing video decoder thread ...")
        self.stream = VideoDecoder(path,
                                   bound,
                                   target_fps,
                                   letterboxed,
                                   new_shape,
                                   color,
                                   stride,
                                   box
                                   )
        self.dc_data: DecodedVideoData = self.stream.read()
        self.stopped = False
        self.delay = 1/target_fps
        print("Video decoder thread initialized")

    def start(self):
        """Start new thread for video decoder"""
        print("Starting video decoder thread ...")
        Thread(target=self.get, args=()).start()
        print("Video decoder thread started")
        return self

    def get(self) -> None:
        """Process next frame and decode"""
        while not self.stopped:
            time.sleep(self.delay)
            if not self.dc_data[0]:
                self.stop()
            else:
                self.dc_data = self.stream.read()

    def read(self) -> DecodedVideoData:
        """Returns latest decoded frame"""
        return self.dc_data

    def stop(self) -> None:
        """Stop the video decoder thread"""
        print("Stopping video decoder thread ...")
        self.stopped = True
        self.stream.stop()
        print("Video decoder thread stopped")

    def __del__(self) -> None:
        print("Stopping video decoder thread ...")
        self.stopped = True
        self.stream.stop()
        print("Video decoder thread stopped")
