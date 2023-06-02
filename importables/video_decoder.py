"""An improved VideoCap class to fetch realtime (even if drop frame) using multiprocessing"""

import os
import time
import queue
from mvextractor.videocap import VideoCap  # pylint: disable=no-name-in-module
from multiprocessing import Queue, Process
from .custom_types import (
    DecodedData,
    FrameRGB,
    FrameType,
    IsDecoderFrameAvailable,
    RawMotionVectors,
    Any
)


class VideoDecoderProcessSpawner:
    """An improved VideoCap class to fetch realtime (even if drop frame) using multiprocessing"""

    def __init__(self, path: str, realtime: bool = False, update_rate: int = 999, buffer_size: int = 0) -> None:
        self.__update_rate: int = update_rate
        self.__delay: float = 1/update_rate
        self.__path: str = path
        self.__run: bool = False
        self.__first_read: bool = False
        self.__realtime: bool = realtime
        self.__queue: Queue[DecodedData] = Queue(maxsize=1 if realtime else buffer_size)  # buffer size will be ignored if realtime == True

    def __refresh(self) -> None:
        count_timeout: int = 0
        video_capturer: VideoCap = VideoCap()
        if not video_capturer.open(self.__path):
            assert False, "ERROR WHILE LOADING " + self.__path
        data: Any = video_capturer.read()
        is_available: IsDecoderFrameAvailable = data[0]
        frame_rgb: FrameRGB = data[1]
        raw_motion_vector: RawMotionVectors = data[2]
        frame_type: FrameType = data[3]
        self.__queue.put(
            (
                is_available,
                frame_rgb,
                raw_motion_vector,
                frame_type
            )
        )  # initialization frame
        timeout_time: int = self.__update_rate*20
        while True:
            start: float = time.perf_counter()
            data: Any = video_capturer.read()
            try:
                is_available: IsDecoderFrameAvailable = data[0]
                frame_rgb: FrameRGB = data[1]
                raw_motion_vector: RawMotionVectors = data[2]
                frame_type: FrameType = data[3]
                self.__queue.put(
                    (
                        is_available,
                        frame_rgb,
                        raw_motion_vector,
                        frame_type
                    ), block=not self.__realtime)
            except queue.Full:
                count_timeout += 1
                if count_timeout >= timeout_time:  # how many frames until it kills itself
                    print("Video capturer timeout, killing process...")
                    video_capturer.release()
                    return
            else:
                count_timeout = 0
            if not data[0]:
                print("Video capturer source empty, killing process...")
                video_capturer.release()
                return
            end: float = time.perf_counter()
            time.sleep(max(0, self.__delay - (end - start)))

    def read(self) -> DecodedData:
        """Pulls frame data from video capturer process and returns it"""
        if self.__run and self.__first_read:
            if not self.__data[0]:  # automatically kill process if source empty
                print("Read: Video capturer source empty, killing process...")
                self.stop(False)
                return self.__data
            try:
                self.__data = self.__queue.get(block=not self.__realtime)
            except queue.Empty:
                _ = ""
        self.__first_read = True
        return self.__data

    def start(self) -> "VideoDecoderProcessSpawner":
        """Spawns new process for video capturer"""
        print("Spawning and initializing video capturer process...")
        self.__process: Process = Process(target=self.__refresh, args=(), daemon=True)
        self.__process.start()
        self.__data: DecodedData = self.__queue.get()
        self.__run = True
        print("Video capturer process started")
        return self

    def stop(self, manual: bool = True) -> "VideoDecoderProcessSpawner":
        """Kill existing process for video capturer"""
        if self.__run:
            print("Stopping video capturer process...")
            if not self.__realtime and manual:  # manual trigger to prevent auto kill recursion
                while self.__data[0]:
                    self.read()
            self.__run = False
            self.__first_read = False
            if self.__process:
                if self.__process.pid:
                    try:
                        os.kill(self.__process.pid, 9)
                    except:
                        pass
            print("Video capturer process stopped")
        return self

    def __del__(self):
        self.stop()
