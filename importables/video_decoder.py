"""Video decoder that decodes video Frames and bounded Motion Vectors images"""

import numpy
from mvextractor.videocap import VideoCap  # pylint: disable=no-name-in-module


MotionVectorData = tuple[bool, numpy.ndarray, numpy.ndarray, str, float]
DecodedVideoData = tuple[bool, numpy.ndarray, numpy.ndarray, numpy.ndarray]


class VideoDecoder:
    """Video decoder that decodes video Frames and bounded Motion Vectors images"""

    def __init__(self, path: str, bound: int) -> None:

        # initiate mvextractor VideoCapture
        self.__video_container: VideoCap = VideoCap()
        if not self.__video_container.open(path):
            assert False, "ERROR WHILE LOADING " + path

        self.__empty = False

        # fetch first frame to get resolution
        last_data: MotionVectorData = self.__video_container.read()
        self.__last_frame: numpy.ndarray = last_data[1]
        self.__res: tuple[int, int] = self.__last_frame.shape[:-1]
        self.__last_flows_x: numpy.ndarray = numpy.ones(self.__res, dtype=numpy.uint8) * 127
        self.__last_flows_y: numpy.ndarray = numpy.ones(self.__res, dtype=numpy.uint8) * 127

        # bound param
        self.set_bound(bound)

    def set_bound(self, bound: int) -> None:
        """Sets top and bottom bound of motion vector conversion (2 x bound)"""
        self.__bound: int = bound
        self.__inverse_rgb_bound_2x: float = 255 / (self.__bound * 2)

    def get_resolution(self) -> tuple[int, int]:
        """Returns video resolution"""
        return self.__res

    def read(self) -> DecodedVideoData:
        """Reads next frame into memory and returns the current"""
        # fetch current frame to return
        available: bool = True
        frame: numpy.ndarray = self.__last_frame
        flows_x: numpy.ndarray = self.__last_flows_x
        flows_y: numpy.ndarray = self.__last_flows_y

        # if no frame left return false on availability
        if self.__empty:
            available: bool = False
            return available, frame, flows_x, flows_y

        # read next frame, if failed then its empty/done
        mv_data: MotionVectorData = self.__video_container.read()
        if not mv_data[0]:
            self.__empty = True
            return available, frame, flows_x, flows_y
        self.__last_frame: numpy.ndarray = mv_data[1]

        # check if there is motion vector, if there is, process
        if len(mv_data[2]) != 0:
            self.__last_flows_x: numpy.ndarray = numpy.zeros(self.__res, dtype=numpy.int16)
            self.__last_flows_y: numpy.ndarray = numpy.zeros(self.__res, dtype=numpy.int16)

            for motion_vector in mv_data[2]:
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

                    self.__last_flows_x[str_y:end_y, str_x:end_x] = motion_x
                    self.__last_flows_y[str_y:end_y, str_x:end_x] = motion_y

            # rescale motion vector with bound to 0 255
            # bound is rephrased
            # translate to 255 first from 2*bound
            # then if < 0 which is lower than bound and > 1 which is higher than bound

            self.__last_flows_x: numpy.ndarray = self.__inverse_rgb_bound_2x * (
                self.__last_flows_x + self.__bound)
            self.__last_flows_y: numpy.ndarray = self.__inverse_rgb_bound_2x * (
                self.__last_flows_y + self.__bound)

            self.__last_flows_x[self.__last_flows_x < 0] = 0
            self.__last_flows_y[self.__last_flows_y < 0] = 0

            self.__last_flows_x[self.__last_flows_x > 255] = 255
            self.__last_flows_y[self.__last_flows_y > 255] = 255

            self.__last_flows_x: numpy.ndarray = self.__last_flows_x.astype(numpy.uint8)
            self.__last_flows_y: numpy.ndarray = self.__last_flows_y.astype(numpy.uint8)

        return available, frame, flows_x, flows_y

    def release(self) -> None:
        """Release memory when not used anymore"""
        self.__video_container.release()

    def __del__(self) -> None:
        self.__video_container.release()
