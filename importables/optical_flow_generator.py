# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
"""Optical Flow Generator generates optical flow frames from two consecutive rgb frames using selected model"""

import cv2
import h5py
import numba
import numpy
import torch
import ptlflow
from torch import Tensor
from numpy import ndarray
from collections import deque
from ptlflow.utils.io_adapter import IOAdapter
from .motion_vector_processor import MotionVectorProcessor
from .utilities import Utilities
from .custom_types import (
    List,
    Tuple,
    FrameRGB,
    ResolutionHW,
    ResolutionHWC,
    OpticalFlowFrame,
)


class OpticalFlowGenerator:
    """Optical Flow Generator generates optical flow frames from two consecutive rgb frames using selected model"""

    def __init__(self,
                 model_type: str,
                 model_pretrained: str,
                 target_size: int = 320,
                 bound: int = 32,
                 raw_optical_flows: bool = False,
                 optical_flow_scale: float = 10,
                 overlap_grid_mode: bool = False,
                 overlap_grid_scale: int = 2
                 ) -> None:

        print(f"Initializing optical flow model ({model_type} -> {model_pretrained})...")

        self.__model_type: str = model_type
        self.__model_pretrained = model_pretrained
        self.__target_size = target_size

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark_limit = 0
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

        # check device capability
        self.__device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__half_capable: bool = False  # only use full float because half causes some problems
        # self.__half_capable: bool = self.__device.type != "cpu"

        # load model
        self.__model: ptlflow.BaseModel = ptlflow.get_model(model_type, model_pretrained)

        # move model
        self.__model = self.__model.to(self.__device)

        # set model precision
        self.__model = self.__model.half() if self.__half_capable else self.__model.float()

        # set model to eval mode
        self.__model.eval()

        # boolean for traced model
        self.__is_traced: bool = False

        # bound param
        self.__optical_flow_scale: float = optical_flow_scale
        self.__raw_optical_flows: bool = raw_optical_flows
        self.__overlap_grid_mode: bool = overlap_grid_mode
        self.__overlap_grid_scale: int = overlap_grid_scale
        self.__bound: int = bound  # bound will be ignored if raw motion vector
        self.__inverse_rgb_2x_bound: float = 255 / (self.__bound * 2)
        self.__half_rgb: int = 128

        print(f"Optical flow model ({self.__model_type} -> {self.__model_pretrained}) initialized")

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def __generate_overlap_grid_core(
        image_list: ndarray,
        scale: int = 2
    ) -> Tuple[ndarray, int, int]:
        image_n, image_h, image_w, image_c = image_list.shape
        grid_h: int = image_h//scale
        grid_w: int = image_w//scale
        h_shift: int = grid_h // 2
        w_shift: int = grid_w // 2
        grid_n: int = (scale**2) + ((scale-1)**2)

        grid_container = numpy.zeros((
            grid_n,
            image_n,
            grid_h,
            grid_w,
            image_c
        ), dtype=image_list.dtype)

        # unshifted
        vertical_split_unshifted: List[ndarray] = numpy.split(image_list, (scale), axis=1)
        for i in numba.prange(scale):
            horizontal_split_unshifted: List[ndarray] = numpy.split(vertical_split_unshifted[i], (scale), axis=2)
            for j in numba.prange(scale):
                idx: int = (i*scale) + j
                grid_container[idx] = horizontal_split_unshifted[j]

        # shifted
        image_shifted: ndarray = image_list[:, h_shift:-h_shift, w_shift:-w_shift]

        vertical_split_shifted: List[ndarray] = numpy.split(image_shifted, (scale-1), axis=1)
        for i in numba.prange(scale-1):
            horizontal_split_shifted: List[ndarray] = numpy.split(vertical_split_shifted[i], (scale-1), axis=2)
            for j in numba.prange(scale-1):
                idx: int = (i*(scale-1)) + j + scale ** 2
                grid_container[idx] = horizontal_split_shifted[j]

        return grid_container, image_h, image_w

    @staticmethod
    def generate_overlap_grid(
        image_list: ndarray,
        scale: int = 2
    ) -> ndarray:

        grids, image_h, image_w = OpticalFlowGenerator.__generate_overlap_grid_core(image_list, scale)

        scaled: List[List[ndarray]] = []
        for img_tuple in grids:
            resized_img_tuple = []
            for img in img_tuple:
                resized_img_tuple.append(cv2.resize(img, (image_w, image_h), interpolation=cv2.INTER_NEAREST))
            scaled.append(resized_img_tuple)

        return numpy.array(scaled)

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def __reconstruct_overlap_grid_core(
        resized_overlap_grid: ndarray,
        overlap_grid_shape: Tuple[int, ...],
        scale: int = 2
    ) -> ndarray:
        grid_n, image_n, image_h, image_w, image_c = overlap_grid_shape
        grid_h: int = image_h // scale
        grid_w: int = image_w // scale
        h_shift: int = grid_h // 2
        w_shift: int = grid_w // 2

        # recreate base image
        base_image: ndarray = numpy.zeros((image_n, image_h, image_w, image_c), dtype=resized_overlap_grid.dtype)

        for i in numba.prange(grid_n):
            if i < scale ** 2:
                # if reading unshifted apply normally
                x_sta: int = (i % scale) * grid_w
                y_sta: int = (i // scale) * grid_h
                x_end: int = x_sta + grid_w
                y_end: int = y_sta + grid_h
                base_image[:, y_sta:y_end, x_sta:x_end] = resized_overlap_grid[i]
            else:
                # if reading shifted, replace if absolute value is bigger
                i_sh: int = i - scale**2
                x_sta: int = (i_sh % (scale-1)) * grid_w + w_shift
                y_sta: int = (i_sh // (scale-1)) * grid_h + h_shift

                for y_off in numba.prange(grid_h):
                    y_tar: int = y_sta + y_off
                    for x_off in numba.prange(grid_w):
                        x_tar: int = x_sta + x_off
                        for img_n in numba.prange(image_n):
                            for chan_n in numba.prange(image_c):
                                curr_val = base_image[img_n, y_tar, x_tar, chan_n]
                                targ_val = resized_overlap_grid[i, img_n, y_off, x_off, chan_n]
                                if abs(targ_val) > abs(curr_val):
                                    base_image[img_n, y_tar, x_tar, chan_n] = targ_val

        return base_image

    @staticmethod
    def reconstruct_overlap_grid(
        overlap_grid: ndarray,
        scale: int = 2
    ):
        _, _, image_h, image_w, _ = overlap_grid.shape
        grid_h: int = image_h // scale
        grid_w: int = image_w // scale

        resized_overlap_grid: List[List[ndarray]] = []
        for img_tuple in overlap_grid:
            resized_img_tuple = [
                cv2.resize(x, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
                for x in img_tuple
            ]
            resized_overlap_grid.append(resized_img_tuple)

        return OpticalFlowGenerator.__reconstruct_overlap_grid_core(numpy.array(resized_overlap_grid), overlap_grid.shape, scale)

    def __first_input(self, image_1: FrameRGB, image_2: FrameRGB) -> None:

        print(f"Warming up optical flow model ({self.__model_type} -> {self.__model_pretrained})...")

        # initialize ioadapter
        self.__ioadapter = IOAdapter(
            self.__model, image_1.shape[:2], cuda=torch.cuda.is_available()
        )

        inputs: dict = self.__ioadapter.prepare_inputs([image_1, image_2])

        with torch.no_grad():
            opt_output: dict = self.__model.forward(inputs)

        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary())
        self.__is_traced = True
        print(f"Optical flow model ({self.__model_type} -> {self.__model_pretrained}) warmed up")

    def forward_once(self, image_1: FrameRGB, image_2: FrameRGB, reverse_scaling: bool = False) -> OpticalFlowFrame:
        """Generate optical flow from two consecutive frames"""

        image_1, original_size, _, _, _, _ = Utilities.letterbox(
            image_1,
            self.__target_size,
            stride=1
        )

        image_2, _, _, _, _, _ = Utilities.letterbox(
            image_2,
            self.__target_size,
            stride=1
        )

        if not self.__is_traced:
            self.__first_input(image_1, image_2)

        inputs: dict = self.__ioadapter.prepare_inputs([image_1, image_2])

        with torch.no_grad():
            opt_output: dict = self.__model.forward(inputs)  # type:ignore

        predictions: dict = self.__ioadapter.unpad_and_unscale(opt_output)

        # times -1 to reverse the orientation
        optical_flows: Tensor = predictions['flows'][0, 0] * -1 * self.__optical_flow_scale
        optical_flows = optical_flows.to(torch.int16)
        optical_flows = optical_flows.permute(1, 2, 0)
        optical_flows_np: OpticalFlowFrame = optical_flows.detach().to("cpu").numpy().copy()

        if not self.__raw_optical_flows:
            optical_flows_np = Utilities.bound_motion_frame(
                optical_flows_np,
                self.__half_rgb,
                self.__inverse_rgb_2x_bound
            )

        if reverse_scaling:
            optical_flows_np = Utilities.unletterbox(
                optical_flows_np,
                original_size,
                0, 0, 0, 0,
                True,
            )

        return optical_flows_np

    def forward(self, image_pair_list: List[Tuple[FrameRGB, FrameRGB]], reverse_scaling: bool = False) -> List[OpticalFlowFrame]:
        """Generate optical flow from list of two consecutive frames"""

        # hacky way from
        # https://github.com/hmorimitsu/ptlflow/issues/28#issuecomment-971726337

        # image_pair_list is currently (B, N, H, W, C)
        # concat to (BxN, H, W, C)
        image_pair_list = numpy.concatenate(image_pair_list, 0)

        # resize to target
        original_size: ResolutionHW = (0, 0)
        resized_image_pair_list: List[FrameRGB] = []
        for image in image_pair_list:
            image_rez, original_size, _, _, _, _ = Utilities.letterbox(
                image,
                self.__target_size,
                stride=1
            )
            resized_image_pair_list.append(image_rez)
        image_pair_list = numpy.array(resized_image_pair_list)

        if not self.__is_traced:
            self.__first_input(image_pair_list[0], image_pair_list[1])

        # Takes shape (BxN, H, W, C) (as a numpy array! torch tensor does not work) and applies padding, etc
        # output is now (1, BxN, C, H, W)
        inputs: dict = self.__ioadapter.prepare_inputs(image_pair_list)

        # convert to batch [B, N, C, W, H]
        input_images: Tensor = inputs["images"][0]  # (BxN, C, H, W)
        list_input_images: List[Tensor] = torch.split(input_images, 2)  # [(B, C, H, W)] x N
        input_images = torch.stack(list_input_images, 0)  # (B, N, C, W, H)
        inputs["images"] = input_images

        with torch.no_grad():
            opt_output: dict = self.__model.forward(inputs)  # type:ignore

        predictions: dict = self.__ioadapter.unpad_and_unscale(opt_output)

        # times -1 to reverse the orientation
        optical_flows: Tensor = predictions['flows'][:, 0] * -1 * self.__optical_flow_scale
        optical_flows = optical_flows.to(torch.int16)
        optical_flows = optical_flows.permute(0, 2, 3, 1)
        optical_flows_np_batch: OpticalFlowFrame = optical_flows.detach().to("cpu").numpy().copy()

        list_optical_flows_np: List[OpticalFlowFrame] = []

        for optical_flows_np in optical_flows_np_batch:
            if not self.__raw_optical_flows:
                optical_flows_np = Utilities.bound_motion_frame(
                    optical_flows_np,
                    self.__half_rgb,
                    self.__inverse_rgb_2x_bound
                )

            if reverse_scaling:
                optical_flows_np = Utilities.unletterbox(
                    optical_flows_np,
                    original_size,
                    0, 0, 0, 0,
                    True,
                )

            list_optical_flows_np.append(optical_flows_np)

        return list_optical_flows_np

    def forward_once_with_overlap_grid(self, image_1: FrameRGB, image_2: FrameRGB, scale: int = 2, reverse_scaling: bool = False) -> OpticalFlowFrame:
        """Generate optical flow from two consecutive frames using overlap grid"""
        image_pair: ndarray = numpy.array((image_1, image_2))
        overlap_grid: ndarray = OpticalFlowGenerator.generate_overlap_grid(image_pair, scale)
        optical_flows: List[OpticalFlowFrame] = self.forward(overlap_grid, reverse_scaling)  # type:ignore
        optical_flows_grid: ndarray = numpy.array(optical_flows)[:, None, ...]
        optical_flows_reconstruct: ndarray = OpticalFlowGenerator.reconstruct_overlap_grid(optical_flows_grid, scale)
        return optical_flows_reconstruct[-1, ...]

    def forward_once_auto(self, image_1: FrameRGB, image_2: FrameRGB, reverse_scaling: bool = False) -> OpticalFlowFrame:
        """Automatically chooses forward once function based on the initial arguments"""
        if self.__overlap_grid_mode:
            return self.forward_once_with_overlap_grid(image_1, image_2, self.__overlap_grid_scale, reverse_scaling)
        return self.forward_once(image_1, image_2)


class OpticalFlowGeneratorMocker(OpticalFlowGenerator):
    """Writes and reads generated optical flow to and from a file, mocking OpticalFlowGenerator behaviour"""

    OPTICAL_FLOWS_PATH = "optical_flows"
    OPTICAL_FLOWS_ARE_RAW_ATTR = "optical_flows_are_raw"
    OPTICAL_FLOWS_BOUND_ATTR = "optical_flows_bound"
    OPTICAL_FLOWS_HWC_ATTR = "optical_flows_hwc"
    OPTICAL_FLOWS_COUNT_ATTR = "optical_flows_count"
    OPTICAL_FLOWS_OVERLAP_GRID_MODE_ATTR = "optical_flows_overlap_grid_mode"
    OPTICAL_FLOWS_OVERLAP_GRID_SCALE_ATTR = "optical_flows_overlap_grid_scale"

    def __init__(
            self,
            h5py_instance: h5py.File,
            bound: int = 32,
            raw_optical_flows: bool = False,
            overlap_grid_mode: bool = False,
            overlap_grid_scale: int = 2,
            *args,
            **kwargs
    ) -> None:
        self.__optical_flows: deque[OpticalFlowFrame] = deque()
        self.__bound: int = bound
        self.__inverse_rgb_2x_bound: float = 255 / (self.__bound * 2)
        self.__half_rgb: int = 128
        self.__raw_optical_flows: bool = raw_optical_flows
        self.__h5py_instance: h5py.File = h5py_instance
        self.__optical_flows_type = True
        self.__optical_flows_hwc: ResolutionHWC = (0, 0, 0)
        self.__overlap_grid_mode: bool = overlap_grid_mode
        self.__overlap_grid_scale: int = overlap_grid_scale

    def load(self) -> bool:
        """Loads optical_flows from file to queues"""
        self.flush()

        if not self.OPTICAL_FLOWS_PATH in self.__h5py_instance:
            return False

        for optical_flows in self.__h5py_instance[self.OPTICAL_FLOWS_PATH][()]:  # type: ignore
            self.__optical_flows.append(optical_flows)

        self.__optical_flows_type = bool(self.__h5py_instance.attrs[self.OPTICAL_FLOWS_ARE_RAW_ATTR])
        self.__optical_flows_hwc = tuple(self.__h5py_instance.attrs[self.OPTICAL_FLOWS_HWC_ATTR])

        return True

    def save(self) -> bool:
        """Saves optical flows in queues to file and flushes the queue"""
        if self.OPTICAL_FLOWS_PATH in self.__h5py_instance:
            del self.__h5py_instance[self.OPTICAL_FLOWS_PATH]

        self.__h5py_instance.create_dataset(name=self.OPTICAL_FLOWS_PATH, data=numpy.array(list(self.__optical_flows)),
                                            compression="gzip",
                                            )

        if not numpy.allclose(self.__optical_flows, self.__h5py_instance[self.OPTICAL_FLOWS_PATH][()]):
            return False

        self.__h5py_instance.attrs[self.OPTICAL_FLOWS_ARE_RAW_ATTR] = self.__raw_optical_flows
        self.__h5py_instance.attrs[self.OPTICAL_FLOWS_BOUND_ATTR] = -1 if self.__raw_optical_flows else self.__bound
        self.__h5py_instance.attrs[self.OPTICAL_FLOWS_HWC_ATTR] = self.__optical_flows[0].shape
        self.__h5py_instance.attrs[self.OPTICAL_FLOWS_COUNT_ATTR] = len(self.__optical_flows)
        self.__h5py_instance.attrs[self.OPTICAL_FLOWS_OVERLAP_GRID_MODE_ATTR] = self.__overlap_grid_mode
        self.__h5py_instance.attrs[self.OPTICAL_FLOWS_OVERLAP_GRID_SCALE_ATTR] = self.__overlap_grid_scale

        self.flush()
        return True

    def append(self, data: OpticalFlowFrame) -> None:
        """Appends optical flows to respective queues"""
        self.__optical_flows.append(data.copy('K'))

    def forward(self, *args, **kwargs) -> List[OpticalFlowFrame]:
        """Pops all optical flows from queue simulates OpticalFlowGenerator's forward"""
        optical_flows: List[OpticalFlowFrame] = []
        while len(self.__optical_flows) > 0:
            optical_flows.append(self.forward_once())
        return optical_flows

    def forward_once(self, *args, **kwargs) -> OpticalFlowFrame:
        """Pops optical flows from queue simulates OpticalFlowGenerator's forward_once"""
        optical_flows: OpticalFlowFrame = numpy.empty((0))
        if len(self.__optical_flows) <= 0:
            if self.__optical_flows_type:
                optical_flows = numpy.zeros(self.__optical_flows_hwc, dtype=numpy.int16)
            else:
                optical_flows = numpy.ones(self.__optical_flows_hwc, dtype=numpy.uint8) * 128
        else:
            optical_flows: OpticalFlowFrame = self.__optical_flows.popleft()

        if not self.__raw_optical_flows and self.__optical_flows_type:
            optical_flows = Utilities.bound_motion_frame(
                optical_flows,
                self.__half_rgb,
                self.__inverse_rgb_2x_bound
            )
        return optical_flows

    def forward_once_with_overlap_grid(self, *args, **kwargs) -> OpticalFlowFrame:
        """Pops optical flows from queue simulates OpticalFlowGenerator's forward_once_with_overlap_grid"""
        return self.forward_once()

    def forward_once_auto(self, *args, **kwargs) -> OpticalFlowFrame:
        """Pops optical flows from queue simulates OpticalFlowGenerator's forward_once_auto"""
        return self.forward_once()

    def flush(self) -> None:
        """Clear all queues"""
        self.__optical_flows.clear()
        self.__optical_flows_type = True
