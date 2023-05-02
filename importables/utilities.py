import cv2
import numba
import numpy
from .custom_types import (
    Tuple,
    ndarray,
    ImageType,
    ColorInput,
    ResolutionHW,
    BoundingBoxXY1XY2,
    FrameOfMotionDataType,
)


class Utilities:
    """Utilities contains all function that reused across different process"""
    def __new__(cls):
        raise TypeError('Utilities is a constant class and cannot be instantiated')

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
    def crop_to_bb(
        image: ImageType,
        bounding_box: BoundingBoxXY1XY2
    ) -> ImageType:
        """Crops image using bounding box"""

        return image[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])].copy()

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
