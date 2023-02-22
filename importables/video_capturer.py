"""An improved VideoCap class to fetch realtime (even if drop frame) using multiprocessing"""

from multiprocessing import Queue, Process, active_children
import queue
from mvextractor.videocap import VideoCap  # pylint: disable=no-name-in-module
import time
from numpy import ndarray
import os

FrameData = tuple[bool, ndarray, ndarray, str, float]


class VideoCapturerProcessSpawner:
    """An improved VideoCap class to fetch realtime (even if drop frame) using multiprocessing"""

    def __init__(self, path: str, sampling_rate: int) -> None:
        self.queue: Queue = Queue(maxsize=1)
        self.sampling_rate = sampling_rate
        self.delay: float = 1/sampling_rate
        self.path: str = path
        self.run: bool = False

    def __refresh(self) -> None:
        count_timeout: int = 0
        cam = VideoCap()
        if not cam.open(self.path):
            assert False, "ERROR WHILE LOADING " + self.path
        self.data: FrameData = cam.read()
        self.queue.put(self.data)
        while True:
            start: float = time.perf_counter()
            self.data = cam.read()
            if not self.data[0]:
                print("Video capturer source empty, killing process...")
                cam.release()
                return
            try:
                self.queue.put_nowait(self.data)
            except queue.Full:
                count_timeout += 1
                if count_timeout >= self.sampling_rate*3:  # how many frames until it kills itself
                    print("Video capturer timeout, killing process...")
                    cam.release()
                    return
            else:
                count_timeout = 0
            end: float = time.perf_counter()
            time.sleep(max(0, self.delay - (end - start)))

    def read(self) -> FrameData:
        """Pulls frame data from video capturer process and returns it"""
        if self.run:
            try:
                self.data: FrameData = self.queue.get_nowait()
            except queue.Empty:
                _ = ""
        return self.data

    def start(self) -> "VideoCapturerProcessSpawner":
        """Spawns new process for video capturer"""
        print("Spawning and initializing video capturer process...")
        self.process: Process = Process(target=self.__refresh, args=(), daemon=True)
        self.process.start()
        self.data: FrameData = self.queue.get()
        self.run = True
        print("Video capturer process started")
        return self

    def stop(self) -> "VideoCapturerProcessSpawner":
        """Kill existing process for video capturer"""
        if self.run:
            print("Stopping video capturer process...")
            self.run = False
            os.kill(self.process.pid, 9)
            print("Video capturer process stopped")
        return self

    def __del__(self):
        self.stop()
