import os
import cv2
import time
import numba
import numpy
import torch
import random
import traceback
import subprocess
import sklearn.metrics
import sklearn.preprocessing
import torchvision.transforms
import torch.utils.tensorboard.writer
import matplotlib.pyplot
from .custom_types import (
    Tuple,
    Union,
    Tensor,
    ndarray,
    ImageType,
    ColorInput,
    ResolutionHW,
    SegmentationMask,
    BoundingBoxXY1XY2,
    FrameOfMotionDataType,
)


class Utilities:
    """Utilities contains all function that reused across different process"""
    def __new__(cls):
        raise TypeError('Utilities is a constant class and cannot be instantiated')
    
    class RamDiskManager:
        """Basic LINUX ramdisk manager manage an EXISTING ramdisk WITH NO CHECKING OR ERROR (cannot create because mount requires super user)"""

        def __init__(self, ramdisk_path) -> None:
            self.__ramdisk_path: str = ramdisk_path
            assert os.path.exists(self.__ramdisk_path), f"""
Ramdisk does not exists (create using superuser before running this)
Using: "sudo mount -t tmpfs -o size=<size> tmpfs {self.__ramdisk_path}"
            """
            self.clear()

        def copy(self, filepath: str) -> str:
            abs_filepath: str = os.path.abspath(filepath)
            subprocess.run(f"cp {abs_filepath} {self.__ramdisk_path}/", shell=True)
            _, filename = os.path.split(abs_filepath)
            return os.path.join(self.__ramdisk_path, filename)

        def clear(self) -> None:
            import subprocess
            subprocess.run(f"rm {self.__ramdisk_path}/*", shell=True)

        def __del__(self) -> None:
            self.clear()

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def bound_motion_frame(
            motion_frame: FrameOfMotionDataType,
            half_rgb: int,
            inverse_rgb_2x_bound: float
    ) -> FrameOfMotionDataType:
        """THIS FUNCTION CHANGES THE ORIGINAL INPUT 
        Static method to rescale raw motion vector data to 0 255 uint8"""
        # rescale motion vector with bound to 0 255
        # bound is refactored to precalculate most of the constant
        # translate to 255 first from 2*bound
        # then if < 0 which is lower than bound and > 1 which is higher than bound
        # from int16 to uint8
        for x in numba.prange(motion_frame.shape[0]):
            for y in numba.prange(motion_frame.shape[1]):
                motion_vector_x = (inverse_rgb_2x_bound * motion_frame[x, y, 0]) + half_rgb
                motion_vector_y = (inverse_rgb_2x_bound * motion_frame[x, y, 1]) + half_rgb

                motion_frame[x, y, 0] = max(min(motion_vector_x, 255), 0)
                motion_frame[x, y, 1] = max(min(motion_vector_y, 255), 0)

        motion_frame = motion_frame.astype(numpy.uint8)

        return motion_frame
    
    @staticmethod
    def bound_motion_frame_tensor(
            motion_frame: Tensor,
            half_rgb: int,
            inverse_rgb_2x_bound: float
    ) -> Tensor:
        """Rescale TENSOR raw motion vector data to 0 255 uint8"""
        motion_frame = (inverse_rgb_2x_bound * motion_frame) + half_rgb
        motion_frame = motion_frame.clamp(0, 255).to(torch.uint8)
        return motion_frame

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def bounding_box_resize(
        bounding_box: BoundingBoxXY1XY2,
        original_size: ResolutionHW,
        target_size: ResolutionHW
    ) -> BoundingBoxXY1XY2:
        """Resizes bounding box from original size (h,w) to target size(h,w).
        Bounding box input must in dtype float64, then result convert to int16"""

        bounding_box = bounding_box.copy()
        width_scale: float = target_size[1] / original_size[1]
        height_scale: float = target_size[0] / original_size[0]
        scale_matrix: ndarray = numpy.array(
            [
                [width_scale, 0],
                [0, height_scale]
            ], dtype=bounding_box.dtype  # type: ignore
        )
        xy_1_pos: ndarray = numpy.array([bounding_box[0], bounding_box[1]])
        xy_2_pos: ndarray = numpy.array([bounding_box[2], bounding_box[3]])
        xy_1_scaled: ndarray = numpy.zeros_like(xy_1_pos)
        xy_2_scaled: ndarray = numpy.zeros_like(xy_2_pos)
        numpy.dot(scale_matrix, xy_1_pos, xy_1_scaled)
        numpy.dot(scale_matrix, xy_2_pos, xy_2_scaled)

        bounding_box[0], bounding_box[1] = xy_1_scaled
        bounding_box[2], bounding_box[3] = xy_2_scaled

        return bounding_box

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def bounding_box_reshape_to_ratio(
        bounding_box: BoundingBoxXY1XY2,
        target_size: ResolutionHW,
        aggressive: bool = True
    ) -> BoundingBoxXY1XY2:
        """Reshapes bounding box to target size's (h,w) ratio.
        Bounding box input must in dtype int16"""

        bounding_box_w: int = bounding_box[2] - bounding_box[0]
        bounding_box_h: int = bounding_box[3] - bounding_box[1]
        bounding_box_size: ndarray = numpy.array([bounding_box_h, bounding_box_w])

        ratios: ndarray = target_size / bounding_box_size
        if ratios[0] == ratios[1]:
            # if ratio already correct
            return bounding_box.astype(numpy.int64)

        pivot_idx: int = int(numpy.argmin(ratios))
        inv_pivot_idx: int = pivot_idx * -1 + 1

        if aggressive:
            # if agresive (ignoring the rounding problem)
            target_scale: int = ratios[pivot_idx]
            scaled_target_size = target_size[inv_pivot_idx] / target_scale
            delta: float = scaled_target_size - bounding_box_size[inv_pivot_idx]
            delta /= 2
            bounding_box[0+pivot_idx] = bounding_box[0+pivot_idx] - int(round(delta - 0.1))
            bounding_box[2+pivot_idx] = bounding_box[2+pivot_idx] + int(round(delta + 0.1))
            if bounding_box[0+pivot_idx] < 0:
                bounding_box[2+pivot_idx] += abs(bounding_box[0+pivot_idx])
                bounding_box[0+pivot_idx] = 0
            if bounding_box[2+pivot_idx] > target_size[inv_pivot_idx]:
                bounding_box[0+pivot_idx] -= (bounding_box[2+pivot_idx] - target_size[inv_pivot_idx])
                bounding_box[2+pivot_idx] = target_size[inv_pivot_idx]

            return bounding_box.astype(numpy.int64)

        if target_size[pivot_idx] / bounding_box_size[pivot_idx] == 2:
            # if pivot size is exacly half
            scaled_target_size = target_size[inv_pivot_idx] // 2
            delta: float = scaled_target_size - bounding_box_size[inv_pivot_idx]
            delta /= 2
            bounding_box[0+pivot_idx] = bounding_box[0+pivot_idx] - int(round(delta - 0.1))
            bounding_box[2+pivot_idx] = bounding_box[2+pivot_idx] + int(round(delta + 0.1))
            if bounding_box[0+pivot_idx] < 0:
                bounding_box[2+pivot_idx] += abs(bounding_box[0+pivot_idx])
                bounding_box[0+pivot_idx] = 0
            if bounding_box[2+pivot_idx] > scaled_target_size:
                bounding_box[0+pivot_idx] -= (bounding_box[2+pivot_idx] - scaled_target_size)
                bounding_box[2+pivot_idx] = scaled_target_size

            return bounding_box.astype(numpy.int64)

        if bounding_box_size[pivot_idx] >= target_size[pivot_idx] // 2:
            # if pivot size more than half (correction will cause stretch)
            return numpy.array((0, 0, target_size[1], target_size[0])).astype(numpy.int64)

        # fix ratio

        # find closest
        offset: int = 1
        while target_size[pivot_idx] % (bounding_box_size[pivot_idx] + offset) != 0:
            offset += 1

        # fix pivot
        delta_pivot: float = offset/2
        bounding_box[0+inv_pivot_idx] = bounding_box[0+inv_pivot_idx] - int(round(delta_pivot - 0.1))
        bounding_box[2+inv_pivot_idx] = bounding_box[2+inv_pivot_idx] + int(round(delta_pivot + 0.1))
        if bounding_box[0+inv_pivot_idx] < 0:
            bounding_box[2+inv_pivot_idx] += abs(bounding_box[0+inv_pivot_idx])
            bounding_box[0+inv_pivot_idx] = 0
        if bounding_box[2+inv_pivot_idx] > target_size[pivot_idx]:
            bounding_box[0+inv_pivot_idx] -= (bounding_box[2+inv_pivot_idx] - target_size[pivot_idx])
            bounding_box[2+inv_pivot_idx] = target_size[pivot_idx]

        target_scale: int = target_size[pivot_idx] // (bounding_box_size[pivot_idx] + offset)

        delta: float = target_size[inv_pivot_idx] // target_scale - bounding_box_size[inv_pivot_idx]
        delta /= 2
        bounding_box[0+pivot_idx] = bounding_box[0+pivot_idx] - int(round(delta - 0.1))
        bounding_box[2+pivot_idx] = bounding_box[2+pivot_idx] + int(round(delta + 0.1))
        if bounding_box[0+pivot_idx] < 0:
            bounding_box[2+pivot_idx] += abs(bounding_box[0+pivot_idx])
            bounding_box[0+pivot_idx] = 0
        if bounding_box[2+pivot_idx] > target_size[inv_pivot_idx]:
            bounding_box[0+pivot_idx] -= (bounding_box[2+pivot_idx] - target_size[inv_pivot_idx])
            bounding_box[2+pivot_idx] = target_size[inv_pivot_idx]

        return bounding_box.astype(numpy.int64)

    @staticmethod
    @numba.njit(fastmath=True, parallel=True)
    def fast_mask_assign(image: ImageType, mask: SegmentationMask, value: Union[Tuple, ndarray]) -> ImageType:
        """THIS FUNCTION CHANGES THE ORIGINAL INPUT 
        Assign 'True' from mask with value"""

        value_np: ndarray = numpy.array(value, dtype=image.dtype)

        for x in numba.prange(image.shape[0]):
            for y in numba.prange(image.shape[1]):
                if mask[x, y]:
                    image[x, y] = value_np

        return image

    @staticmethod
    def crop_to_bb(
        image: ImageType,
        bounding_box: BoundingBoxXY1XY2
    ) -> ImageType:
        """Crops image using bounding box"""
        return image[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])].copy()
    
    @staticmethod
    def crop_to_bb_tensor(
        image: Tensor,
        bounding_box: BoundingBoxXY1XY2
    ) -> Tensor:
        """Crops image using bounding box"""
        return torch.clone(image[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])])

    @staticmethod
    def crop_to_bb_and_resize(
        image: ImageType,
        bounding_box: BoundingBoxXY1XY2,
        original_size: ResolutionHW,
        target_size: ResolutionHW = (-1, -1),
        aggressive: bool = True
    ) -> ImageType:
        """Crops image using bounding box, and resizes the image back to original size (h,w) or target size (h,w)"""

        image_size: Tuple[int, int] = image.shape[:2]

        if target_size == (-1, -1):
            target_size = numpy.array(image_size)
        else:
            target_size = numpy.array(target_size)

        bounding_box = Utilities.bounding_box_reshape_to_ratio(bounding_box.astype(numpy.int16), target_size, aggressive)

        if image_size != original_size:
            bounding_box = Utilities.bounding_box_resize(bounding_box.astype(numpy.float64), original_size, image_size).astype(numpy.int16)

        cropped_image: ImageType = Utilities.crop_to_bb(image, bounding_box)
        bounding_box_w: int = bounding_box[2] - bounding_box[0]
        bounding_box_h: int = bounding_box[3] - bounding_box[1]
        bounding_box_size: ndarray = numpy.array([bounding_box_h, bounding_box_w])
        if any(bounding_box_size != target_size):
            return cv2.resize(cropped_image, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

        return cropped_image

    @staticmethod
    def __letterbox_core(
        shape: ResolutionHW,
        target_size: int,
        stride: int,
        box: bool,
    ) -> tuple[ResolutionHW, int, int, int, int]:
        # Scale ratio (new / old)
        r = max(shape)
        r = target_size / r

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = target_size - new_unpad[0], target_size - new_unpad[1]  # wh padding
        if not box:
            dw, dh = dw % stride, dh % stride  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        return new_unpad, top, bottom, left, right

    @staticmethod
    def letterbox(
            image: ImageType,
            target_size: int = 320,
            color: ColorInput = (114, 114, 114),
            stride: int = 32,
            box: bool = False
    ) -> tuple[ImageType, ResolutionHW, int, int, int, int]:
        """Resize and pad image while meeting stride-multiple constraints.
        Returns letterboxed_image_numpy_array, original_size,
        top_pad, bottom_pad, left_pad, right_pad"""

        original_size: ResolutionHW = image.shape[:2]  # current shape [height, width]

        new_unpad, top, bottom, left, right = Utilities.__letterbox_core(
            original_size, target_size, stride, box
        )

        if original_size[::-1] != new_unpad:  # resize
            image = cv2.resize(image, tuple(new_unpad), interpolation=cv2.INTER_NEAREST)  # switch to NEAREST for faster and sharper interp
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return image, original_size, top, bottom, left, right
    
    @staticmethod
    def letterbox_tensor(
            image: Tensor,
            target_size: int = 320,
            color: ColorInput = (114, 114, 114),
            stride: int = 32,
            box: bool = False
    ) -> tuple[Tensor, ResolutionHW, int, int, int, int]:
        """Resize and pad image while meeting stride-multiple constraints.
        Returns letterboxed_image_tensor(...HW), original_size,
        top_pad, bottom_pad, left_pad, right_pad"""

        original_size: ResolutionHW = image.shape[-2:]  # current shape [height, width]

        new_unpad, top, bottom, left, right = Utilities.__letterbox_core(
            original_size, target_size, stride, box
        )

        if original_size[::-1] != new_unpad:  # resize
            image = torchvision.transforms.Resize(new_unpad[::-1],interpolation=torchvision.transforms.InterpolationMode.NEAREST)(image)
        image = torchvision.transforms.Pad((left, top, right, bottom), fill=color[0])(image) 
        return image, original_size, top, bottom, left, right

    @staticmethod
    def unletterbox(
        image: ImageType,
        original_size: ResolutionHW,
        top_pad: int,
        bottom_pad: int,
        left_pad: int,
        right_pad: int,
        resize_to_original: bool,
    ) -> ImageType:
        """Reverses letterbox process"""

        is_boolean = False
        if image.dtype == bool:
            is_boolean = True
            image = image*255
        image = image[top_pad:image.shape[0] - bottom_pad, left_pad:image.shape[1] - right_pad]
        if resize_to_original:
            image = cv2.resize(image, tuple(original_size[::-1]), interpolation=cv2.INTER_NEAREST)  # switch to NEAREST for faster and sharper interp
        if is_boolean:
            image = image > 127  # type: ignore
        return image

    @staticmethod
    def unletterbox_bounding_box(
        bounding_box: BoundingBoxXY1XY2,
        letterboxed_size: ResolutionHW,
        original_size: ResolutionHW,
        top_pad: int,
        bottom_pad: int,
        left_pad: int,
        right_pad: int,
        resize_to_original: bool,
    ) -> BoundingBoxXY1XY2:
        """Reverses letterbox process to bounding box"""

        letterboxed_size = numpy.array(
            (
                letterboxed_size[0] - (top_pad + bottom_pad),
                letterboxed_size[1] - (left_pad + right_pad)
            )
        )
        bounding_box = bounding_box.copy()
        bounding_box[0] = max(bounding_box[0] - left_pad, 0)
        bounding_box[1] = max(bounding_box[1] - top_pad, 0)
        bounding_box[2] = min(bounding_box[2] - left_pad, letterboxed_size[1])
        bounding_box[3] = min(bounding_box[3] - top_pad, letterboxed_size[0])
        if resize_to_original:
            bounding_box = Utilities.bounding_box_resize(bounding_box.astype(numpy.float64), letterboxed_size, original_size).astype(numpy.int16)
        return bounding_box

    @staticmethod
    def set_all_seed(seed_bytes: bytes) -> None:
        """Set random generator seed to seed_bytes"""
        random.seed(seed_bytes)
        numpy.random.seed(int.from_bytes(seed_bytes[:4], "big"))
        torch.manual_seed(int.from_bytes(seed_bytes[:4], "big"))

    @staticmethod
    def write_all_summary_iteration(
        summary_writer: torch.utils.tensorboard.writer.SummaryWriter,
        y_true: ndarray,
        y_pred: ndarray,
        loss: float,
        fps: float,
        encoder: sklearn.preprocessing.LabelEncoder,
        dataset_map: dict,
        step: int,
        epoch: int,
        memory_average: dict,
        is_train: bool = True,
        retry_max: int = 10,
    ) -> Tuple[dict, torch.utils.tensorboard.writer.SummaryWriter]:
        
        if epoch not in memory_average:
            memory_average[epoch] = {
                "Step": 0,
                "Loss": 0,
                "FPS": 0,
                "Accuracy": {
                    "Average": 0,
                    "Weighted Average": 0,
                },
                "Precision": {
                    "Average": 0,
                    "Weighted Average": 0,
                },
                "Recall": {
                    "Average": 0,
                    "Weighted Average": 0,
                },
                "F1 Score": {
                    "Average": 0,
                    "Weighted Average": 0,
                },
                "Support": {
                    "Average": 0,
                    "Weighted Average": 0,
                },
            }

            for i in range(len(encoder.classes_)):
                target_label = dataset_map[encoder.classes_[i]]
                memory_average[epoch]["Accuracy"][target_label] = [0, 0]
                memory_average[epoch]["Precision"][target_label] = [0, 0]
                memory_average[epoch]["Recall"][target_label] = [0, 0]
                memory_average[epoch]["F1 Score"][target_label] = [0, 0]
                memory_average[epoch]["Support"][target_label] = 0

            if epoch > 0:
                if epoch-1 in memory_average:
                    del memory_average[epoch-1]

        memory_average[epoch]["Step"] += 1
        current_step = memory_average[epoch]["Step"]


        current_mode = "Train"
        if not is_train:
            current_mode = "Test"

        accuracies_dict = {}
        precisions_dict = {}
        recalls_dict = {}
        f1_dict = {}
        support_dict = {}

        accuracies_dict_avg = {}
        precisions_dict_avg = {}
        recalls_dict_avg = {}
        f1_dict_avg = {}
        support_dict_avg = {}

        y_true_labelized = encoder.inverse_transform(y_true[..., None])
        y_pred_labelized = encoder.inverse_transform(y_pred[..., None])

        confusion_matrix: ndarray = sklearn.metrics.confusion_matrix(y_true_labelized, y_pred_labelized, labels=encoder.classes_)

        if "Confusion Matrix" not in memory_average[epoch]:
            memory_average[epoch]["Confusion Matrix"] = confusion_matrix
        else:
            memory_average[epoch]["Confusion Matrix"] += confusion_matrix

        accuracies_dict["Average"] = sklearn.metrics.accuracy_score(y_true, y_pred)
        memory_average[epoch]["Accuracy"]["Average"] += accuracies_dict["Average"]
        accuracies_dict_avg["Average"] = memory_average[epoch]["Accuracy"]["Average"] / current_step

        accuracies_dict["Weighted Average"] = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
        memory_average[epoch]["Accuracy"]["Weighted Average"] += accuracies_dict["Weighted Average"]
        accuracies_dict_avg["Weighted Average"] = memory_average[epoch]["Accuracy"]["Weighted Average"] / current_step

        confusion_matrix_norm: ndarray = sklearn.metrics.confusion_matrix(y_true_labelized, y_pred_labelized, normalize="true", labels=encoder.classes_)
        for i in range(len(encoder.classes_)):
            target_label = dataset_map[encoder.classes_[i]]
            if sum(confusion_matrix_norm[i]) > 0:
                accuracies_dict[target_label] = confusion_matrix_norm[i][i]
                memory_average[epoch]["Accuracy"][target_label][0] += accuracies_dict[target_label]
                memory_average[epoch]["Accuracy"][target_label][1] += 1

            if (data_len := memory_average[epoch]["Accuracy"][target_label][1]) > 0:
                accuracies_dict_avg[target_label] = memory_average[epoch]["Accuracy"][target_label][0] / data_len

        classification_report: dict = sklearn.metrics.classification_report(y_true_labelized, y_pred_labelized, labels=encoder.classes_, output_dict=True)  # type: ignore
        for k, v in classification_report.items():

            if k in ("accuracy", "micro avg"):
                continue

            if k == "macro avg":
                precisions_dict["Average"] = v["precision"]
                recalls_dict["Average"] = v["recall"]
                f1_dict["Average"] = v["f1-score"]
                support_dict["Average"] = v["support"]
                memory_average[epoch]["Precision"]["Average"] += v["precision"]
                memory_average[epoch]["Recall"]["Average"] += v["recall"]
                memory_average[epoch]["F1 Score"]["Average"] += v["f1-score"]
                memory_average[epoch]["Support"]["Average"] += v["support"]
                precisions_dict_avg["Average"] = memory_average[epoch]["Precision"]["Average"] / current_step
                recalls_dict_avg["Average"] = memory_average[epoch]["Recall"]["Average"] / current_step
                f1_dict_avg["Average"] = memory_average[epoch]["F1 Score"]["Average"] / current_step
                support_dict_avg["Average"] = memory_average[epoch]["Support"]["Average"]
                continue

            if k == "weighted avg":
                precisions_dict["Weighted Average"] = v["precision"]
                recalls_dict["Weighted Average"] = v["recall"]
                f1_dict["Weighted Average"] = v["f1-score"]
                support_dict["Weighted Average"] = v["support"]
                memory_average[epoch]["Precision"]["Weighted Average"] += v["precision"]
                memory_average[epoch]["Recall"]["Weighted Average"] += v["recall"]
                memory_average[epoch]["F1 Score"]["Weighted Average"] += v["f1-score"]
                memory_average[epoch]["Support"]["Weighted Average"] += v["support"]
                precisions_dict_avg["Weighted Average"] = memory_average[epoch]["Precision"]["Weighted Average"] / current_step
                recalls_dict_avg["Weighted Average"] = memory_average[epoch]["Recall"]["Weighted Average"] / current_step
                f1_dict_avg["Weighted Average"] = memory_average[epoch]["F1 Score"]["Weighted Average"] / current_step
                support_dict_avg["Weighted Average"] = memory_average[epoch]["Support"]["Weighted Average"]
                continue

            if v["support"] != 0:
                precisions_dict[dataset_map[int(k)]] = v["precision"]
                recalls_dict[dataset_map[int(k)]] = v["recall"]
                f1_dict[dataset_map[int(k)]] = v["f1-score"]
                support_dict[dataset_map[int(k)]] = v["support"]
                memory_average[epoch]["Precision"][dataset_map[int(k)]][0] += v["precision"]
                memory_average[epoch]["Precision"][dataset_map[int(k)]][1] += 1
                memory_average[epoch]["Recall"][dataset_map[int(k)]][0] += v["recall"]
                memory_average[epoch]["Recall"][dataset_map[int(k)]][1] += 1
                memory_average[epoch]["F1 Score"][dataset_map[int(k)]][0] += v["f1-score"]
                memory_average[epoch]["F1 Score"][dataset_map[int(k)]][1] += 1
                memory_average[epoch]["Support"][dataset_map[int(k)]] += v["support"]

            if (data_len := memory_average[epoch]["Precision"][dataset_map[int(k)]][1]) > 0:
                precisions_dict_avg[dataset_map[int(k)]] = memory_average[epoch]["Precision"][dataset_map[int(k)]][0] / data_len
            if (data_len := memory_average[epoch]["Recall"][dataset_map[int(k)]][1]) > 0:
                recalls_dict_avg[dataset_map[int(k)]] = memory_average[epoch]["Recall"][dataset_map[int(k)]][0] / data_len
            if (data_len := memory_average[epoch]["F1 Score"][dataset_map[int(k)]][1]) > 0:
                f1_dict_avg[dataset_map[int(k)]] = memory_average[epoch]["F1 Score"][dataset_map[int(k)]][0] / data_len
            support_dict_avg[dataset_map[int(k)]] = memory_average[epoch]["Support"][dataset_map[int(k)]]

        memory_average[epoch]["Loss"] += loss
        memory_average[epoch]["FPS"] += fps

        retry_count = 0

        while True:
            try:
                # iterations
                summary_writer.add_scalar(f"Loss/{current_mode}/Iteration", loss, step)
                summary_writer.add_scalar(f"FPS/{current_mode}/Iteration", fps, step)
                summary_writer.add_scalar(f"Accuracy/Average/{current_mode}/Iteration", accuracies_dict["Average"], step)
                summary_writer.add_scalar(f"Precision/Average/{current_mode}/Iteration", precisions_dict["Average"], step)
                summary_writer.add_scalar(f"Recall/Average/{current_mode}/Iteration", recalls_dict["Average"], step)
                summary_writer.add_scalar(f"F1 Score/Average/{current_mode}/Iteration", f1_dict["Average"], step)
                summary_writer.add_scalar(f"Support/Average/{current_mode}/Iteration", support_dict["Average"], step)
                summary_writer.add_scalar(f"Accuracy/Weighted Average/{current_mode}/Iteration", accuracies_dict["Weighted Average"], step)
                summary_writer.add_scalar(f"Precision/Weighted Average/{current_mode}/Iteration", precisions_dict["Weighted Average"], step)
                summary_writer.add_scalar(f"Recall/Weighted Average/{current_mode}/Iteration", recalls_dict["Weighted Average"], step)
                summary_writer.add_scalar(f"F1 Score/Weighted Average/{current_mode}/Iteration", f1_dict["Weighted Average"], step)
                summary_writer.add_scalar(f"Support/Weighted Average/{current_mode}/Iteration", support_dict["Weighted Average"], step)

                for data in ["Average", "Weighted Average"]:
                    del accuracies_dict[data]
                    del precisions_dict[data]
                    del recalls_dict[data]
                    del f1_dict[data]
                    del support_dict[data]

                # summary_writer.add_scalars(f"Accuracy/Classes/{current_mode}/Iteration", accuracies_dict, step)
                # summary_writer.add_scalars(f"Precision/Classes/{current_mode}/Iteration", precisions_dict, step)
                # summary_writer.add_scalars(f"Recall/Classes/{current_mode}/Iteration", recalls_dict, step)
                # summary_writer.add_scalars(f"F1 Score/Classes/{current_mode}/Iteration", f1_dict, step)
                # summary_writer.add_scalars(f"Support/Classes/{current_mode}/Iteration", support_dict, step)

                # epochs

                summary_writer.add_scalar(f"Loss/{current_mode}", memory_average[epoch]["Loss"]/current_step, epoch)
                summary_writer.add_scalar(f"FPS/{current_mode}", memory_average[epoch]["FPS"]/current_step, epoch)
                summary_writer.add_scalar(f"Accuracy/Average/{current_mode}", accuracies_dict_avg["Average"], epoch)
                summary_writer.add_scalar(f"Precision/Average/{current_mode}", precisions_dict_avg["Average"], epoch)
                summary_writer.add_scalar(f"Recall/Average/{current_mode}", recalls_dict_avg["Average"], epoch)
                summary_writer.add_scalar(f"F1 Score/Average/{current_mode}", f1_dict_avg["Average"], epoch)
                summary_writer.add_scalar(f"Support/Average/{current_mode}", support_dict_avg["Average"], epoch)
                summary_writer.add_scalar(f"Accuracy/Weighted Average/{current_mode}", accuracies_dict_avg["Weighted Average"], epoch)
                summary_writer.add_scalar(f"Precision/Weighted Average/{current_mode}", precisions_dict_avg["Weighted Average"], epoch)
                summary_writer.add_scalar(f"Recall/Weighted Average/{current_mode}", recalls_dict_avg["Weighted Average"], epoch)
                summary_writer.add_scalar(f"F1 Score/Weighted Average/{current_mode}", f1_dict_avg["Weighted Average"], epoch)
                summary_writer.add_scalar(f"Support/Weighted Average/{current_mode}", support_dict_avg["Weighted Average"], epoch)

                for data in ["Average", "Weighted Average"]:
                    del accuracies_dict_avg[data]
                    del precisions_dict_avg[data]
                    del recalls_dict_avg[data]
                    del f1_dict_avg[data]
                    del support_dict_avg[data]

                # summary_writer.add_scalars(f"Accuracy/Classes/{current_mode}", accuracies_dict_avg, epoch)
                # summary_writer.add_scalars(f"Precision/Classes/{current_mode}", precisions_dict_avg, epoch)
                # summary_writer.add_scalars(f"Recall/Classes/{current_mode}", recalls_dict_avg, epoch)
                # summary_writer.add_scalars(f"F1 Score/Classes/{current_mode}", f1_dict_avg, epoch)
                # summary_writer.add_scalars(f"Support/Classes/{current_mode}", support_dict_avg, epoch)

                # summary_writer.add_figure(f"Cumulative Confusion Matrix/{current_mode}", confusion_matrix_cum_display.figure_, epoch)
                # summary_writer.add_figure(f"Truth-Normalized (Class Accuracy) Cumulative Confusion Matrix/{current_mode}", confusion_matrix_cum_norm_display.figure_, epoch)

            except:
                retry_count += 1
                if retry_count <= retry_max:
                    traceback.print_exc()
                    print(f"Logging failed, restarting SummaryWriter ({retry_count}/{retry_max})")
                    log_dir: str = summary_writer.get_logdir()
                    summary_writer.close()
                    summary_writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir)
                else:
                    print(f"Logging failed and max retry is reached, skipping ...")
                    break
            else:
                break

        return memory_average, summary_writer

    @staticmethod
    def write_confusion_matrix_epoch(
        summary_writer: torch.utils.tensorboard.writer.SummaryWriter,
        encoder: sklearn.preprocessing.LabelEncoder,
        dataset_map: dict,
        epoch: int,
        memory_average: dict,
        is_train: bool = True,
        retry_max: int = 10,
    ):
        
        current_mode = "Train"
        if not is_train:
            current_mode = "Test"

        matplotlib.rc('font', size=6)

        _, ax = matplotlib.pyplot.subplots(figsize=(10,10))
        confusion_matrix_cum_display: sklearn.metrics.ConfusionMatrixDisplay = sklearn.metrics.ConfusionMatrixDisplay(
            memory_average[epoch]["Confusion Matrix"], display_labels=[dataset_map[x] for x in encoder.classes_]).plot(ax=ax)
        confusion_matrix_cum_display.ax_.set_xticklabels(confusion_matrix_cum_display.ax_.get_xticklabels(), rotation=45, ha='right')

        confusion_matrix_cum: ndarray = memory_average[epoch]["Confusion Matrix"]
        _, ax = matplotlib.pyplot.subplots(figsize=(10,10))
        confusion_matrix_cum_norm_display: sklearn.metrics.ConfusionMatrixDisplay = sklearn.metrics.ConfusionMatrixDisplay(
            confusion_matrix_cum/confusion_matrix_cum.sum(axis=1)[..., None], display_labels=[dataset_map[x] for x in encoder.classes_]).plot(ax=ax)
        confusion_matrix_cum_norm_display.ax_.set_xticklabels(confusion_matrix_cum_norm_display.ax_.get_xticklabels(), rotation=45, ha='right')

        retry_count = 0

        while True:
            try:

                summary_writer.add_figure(f"Cumulative Confusion Matrix/{current_mode}", confusion_matrix_cum_display.figure_, epoch)
                summary_writer.add_figure(f"Truth-Normalized (Class Accuracy) Cumulative Confusion Matrix/{current_mode}", confusion_matrix_cum_norm_display.figure_, epoch)

            except:
                retry_count += 1
                if retry_count <= retry_max:
                    traceback.print_exc()
                    print(f"Confusion Matrix Logging failed, restarting SummaryWriter ({retry_count}/{retry_max})")
                    log_dir: str = summary_writer.get_logdir()
                    summary_writer.close()
                    summary_writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir)
                else:
                    print(f"Confusion Matrix Logging failed and max retry is reached, skipping ...")
                    break
            else:
                break

        return memory_average, summary_writer