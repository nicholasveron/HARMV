# pyright: reportPrivateImportUsage=false
"""Mask Generator generates peoples mask using YOLOv7 - Mask from bgr frames"""
import os
import h5py
import yaml
import numpy
import torch
import nebullvm
import speedster
import traceback
import torchvision
from torch import Tensor
from numpy import ndarray
from collections import deque
from detectron2.structures import Boxes
from detectron2.modeling.poolers import ROIPooler
from detectron2.layers import paste_masks_in_image
from .utilities import Utilities
from .custom_types import (
    Tuple,
    Union,
    ColorRGB,
    FrameRGB,
    MaskOnlyData,
    ResolutionHW,
    IsHumanDetected,
    SegmentationMask,
    BoundingBoxXY1XY2,
    MaskWithMostCenterBoundingBoxData,
)


class MaskGenerator:
    """Mask Generator generates peoples mask using YOLOv7 - Mask from bgr frames"""

    def __init__(self,
                 weight_path: str,
                 hyperparameter_path: str,
                 confidence_threshold: float,
                 iou_threshold: float,
                 target_size: int = 320,
                 resize_result_to_original: bool = False,
                 optimize_model: bool = True,
                 target_device: Union[str, torch.device, None] = None,
                 letterbox_stride: int = 32,
                 letterbox_box_mode: bool = False,
                 letterbox_color: ColorRGB = (114, 114, 114),
                 bounding_box_grouping_range_scale: float = 1,
                 bounding_box_no_merge: bool = False,
                 flip_bgr_rgb: bool = True,
                 ) -> None:

        print("Initializing mask generator...")

        # check device capability
        self.__device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if target_device:
            if isinstance(target_device, str):
                self.__device = torch.device(target_device)
                print("target", target_device, self.__device)
            else:
                self.__device = target_device
        self.__half_capable: bool = self.__device.type == "cuda"
        self.__dtype = torch.float16 if self.__half_capable else torch.float32

        # load model
        self.__model = torch.load(weight_path, map_location={
            "cpu": self.__device.type,
            "cuda:0": self.__device.type,
            "cuda:1": self.__device.type,
            "cuda:2": self.__device.type,
        })['model']

        for m in self.__model.model:
            if isinstance(m, torch.nn.Upsample):
                setattr(m, "recompute_scale_factor", None)

        # set model precision
        self.__model = self.__model.to(device=self.__device, dtype=self.__dtype)

        # set model to eval mode
        self.__model.eval()

        # load hyperparameters
        with open(hyperparameter_path) as hyp_file:  # pylint: disable=unspecified-encoding
            self.__hyperparameters: dict = yaml.load(hyp_file, Loader=yaml.FullLoader)

        # initialize pooler
        self.__pooler = ROIPooler(
            output_size=self.__hyperparameters['mask_resolution'],
            scales=(self.__model.pooler_scale,),
            sampling_ratio=1,
            pooler_type='ROIAlignV2',
            canonical_level=2
        )

        # boolean for traced model
        self.__is_traced = False

        # other parameters
        self.__confidence_threshold: float = confidence_threshold
        self.__iou_threshold: float = iou_threshold
        self.__optimize_model: bool = optimize_model
        self.__flip_bgr_rgb: bool = flip_bgr_rgb

        # letterbox params
        self.__target_size: int = target_size
        self.__resize_result_to_original: bool = resize_result_to_original
        self.__letterbox_stride: int = letterbox_stride
        self.__letterbox_box_mode: bool = letterbox_box_mode
        self.__letterbox_color: ColorRGB = letterbox_color
        self.__bounding_box_grouping_range_scale: float = bounding_box_grouping_range_scale
        self.__bounding_box_no_merge: bool = bounding_box_no_merge

        # bounding box params

        print("Mask generator initialized")

    @staticmethod
    @torch.jit.script
    def __unpack_inference(x: Tensor, bbox_w: int, bbox_h: int, bbox_pix: int) -> Tuple[Tensor, Tensor, Tensor]:
        inf: Tensor = x[:, :, :85]
        attn: Tensor = x[:, :, 85:1065]
        bases: Tensor = x[:, :, 1065:]

        bases = bases.flatten()
        bases = bases[:bbox_pix]
        bases = bases.reshape(1, 5, bbox_h, bbox_w)

        return inf, attn, bases

    @staticmethod
    @torch.jit.script
    def __xywh2xyxy(x: Tensor) -> Tensor:
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y: Tensor = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    @staticmethod
    @torch.jit.script
    def __merge_bases(rois: Tensor, coeffs: Tensor, attn_r: int, num_b: int) -> Tensor:
        # merge predictions
        N, _, H, W = rois.size()
        if coeffs.dim() != 4:
            coeffs = coeffs.view(N, num_b, attn_r, attn_r)
        coeffs = torch.nn.functional.interpolate(coeffs, (H, W),
                                                 mode="bilinear").softmax(dim=1)
        masks_preds: Tensor = (rois * coeffs).sum(dim=1)
        return masks_preds

    @staticmethod
    def __loopable_nms_conf_people_only(
        x: Tensor,
        xc: Tensor,
        attn: Tensor,
        base: Tensor,
        attn_res: int,
        num_base: int,
        pooler: ROIPooler,
        conf_thres: float,
        iou_thres: float,
        multi_label: bool,
        max_wh: int,
        max_det: int
    ) -> Tuple[bool, Tensor, Tensor]:
        output: Tensor = torch.empty((0), device=x.device, dtype=x.dtype)
        output_mask: Tensor = torch.empty((0), device=x.device, dtype=x.dtype)

        x = x[xc]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            return False, output, output_mask

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box: Tensor = MaskGenerator.__xywh2xyxy(x[:, :4])

        a: Tensor = attn[xc]

        bboxes: Boxes = Boxes(box)

        base_li: list[Tensor] = [base[None]]
        bbox_li: list[Boxes] = [bboxes]
        pooled_bases: Tensor = pooler(base_li, bbox_li)

        pred_masks: Tensor = MaskGenerator.__merge_bases(pooled_bases, a, attn_res, num_base).view(a.shape[0], -1).sigmoid()

        temp: Tensor = pred_masks.clone()
        temp[temp < 0.5] = 1 - temp[temp < 0.5]
        log_score: Tensor = torch.log(temp)
        mean_score: Tensor = log_score.mean(-1, True)
        mask_score: Tensor = torch.exp(mean_score)

        x[:, 5:] *= x[:, 4:5] * mask_score

        if multi_label:
            conf_label: Tensor = x[:, 5:] > conf_thres
            conf_label: Tensor = conf_label.nonzero()
            conf_label: Tensor = conf_label.T
            i, j = conf_label[0], conf_label[1]
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            mask_score = mask_score[i]
            if attn is not None:
                pred_masks = pred_masks[i]
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        filter: Tensor = x[:, 5:6] == 0  # filters human only (0 index), chair is 56
        filter = filter.any(1)
        x = x[filter]
        pred_masks = pred_masks[filter]

        # If none remain process next image
        n: int = x.shape[0]  # number of boxes
        if not n:
            return False, output, output_mask

        # Batched NMS
        c: Tensor = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i: Tensor = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output: Tensor = x[i]
        if attn is not None:
            output_mask: Tensor = pred_masks[i]

        return True, output, output_mask

    @staticmethod
    def __nms_conf_people_only(
        prediction: Tensor,
        attn: Tensor,
        bases: Tensor,
        attn_res: int,
        num_base: int,
        pooler: ROIPooler,
        conf_thres: float = 0.1,
        iou_thres: float = 0.6
    ) -> Tuple[list[bool], list[Tensor], list[Tensor]]:
        nc: int = prediction[0].shape[1] - 5  # number of classes
        xc: Tensor = prediction[..., 4] > conf_thres  # candidates
        # Settings
        max_wh: int = 4096  # (pixels) minimum and maximum box width and height
        max_det: int = 300  # maximum number of detections per image
        multi_label: bool = nc > 1  # multiple labels per box (adds 0.5ms/img)

        found: list[bool] = []
        output: list[Tensor] = []
        output_mask: list[Tensor] = []

        futures: list[torch.jit.Future[Tuple[bool, Tensor, Tensor]]] = []
        for xi in range(len(prediction)):  # image index, image inference
            futures.append(torch.jit.fork(MaskGenerator.__loopable_nms_conf_people_only,
                                          prediction[xi],
                                          xc[xi],
                                          attn[xi],
                                          bases[xi],
                                          attn_res,
                                          num_base,
                                          pooler,
                                          conf_thres,
                                          iou_thres,
                                          multi_label,
                                          max_wh,
                                          max_det
                                          ))  # type: ignore

        for future in futures:
            found_s, output_s, output_mask_s = torch.jit.wait(future)
            found.append(found_s)
            output.append(output_s)
            output_mask.append(output_mask_s)

        return found, output, output_mask

    @staticmethod
    def __nms_conf_people_only_once(
        prediction: Tensor,
        attn: Tensor,
        bases: Tensor,
        attn_res: int,
        num_base: int,
        pooler: ROIPooler,
        conf_thres: float = 0.1,
        iou_thres: float = 0.6
    ) -> Tuple[bool, Tensor, Tensor]:
        nc: int = prediction[0].shape[1] - 5  # number of classes
        xc: Tensor = prediction[..., 4] > conf_thres  # candidates
        # Settings
        max_wh: int = 4096  # (pixels) minimum and maximum box width and height
        max_det: int = 300  # maximum number of detections per image
        multi_label: bool = nc > 1  # multiple labels per box (adds 0.5ms/img)

        found_s, output_s, output_mask_s = MaskGenerator.__loopable_nms_conf_people_only(
            prediction[0],
            xc[0],
            attn[0],
            bases[0],
            attn_res,
            num_base,
            pooler,
            conf_thres,
            iou_thres,
            multi_label,
            max_wh,
            max_det
        )

        return found_s, output_s, output_mask_s

    @staticmethod
    def __find_most_center_bb(
        bounding_boxes: Boxes,
        frame_size: ResolutionHW,
        grouping_range_scale: float = 1,
        no_merge_bounding_box=False
    ) -> Tuple[Boxes, Boxes]:
        # get shortest distance to middle
        center_x: int = frame_size[1]//2
        center_y: int = frame_size[0]//2
        center_np: Tensor = torch.tensor([center_x, center_y], device=bounding_boxes.device)
        bbs_center: Tensor = bounding_boxes.get_centers()
        distances: Tensor = torch.sqrt(((bbs_center-center_np)**2).sum(dim=1))
        shortest_idx = int(torch.argmin(distances).cpu().numpy())
        center_bounding_box: Boxes = bounding_boxes[shortest_idx]
        bounding_boxes = bounding_boxes[torch.arange(len(bounding_boxes)) != shortest_idx]
        if len(bounding_boxes) > 0 and not no_merge_bounding_box:
            center_bounding_box_center: Tensor = center_bounding_box.get_centers()[0]
            center_bounding_box_half_size: Tensor = center_bounding_box_center - center_bounding_box.tensor[-1, :2]
            max_distance: Tensor = grouping_range_scale * (center_bounding_box.area()[0] ** (1/2))

            bbs_center = bounding_boxes.get_centers()
            bbs_half_size: Tensor = bbs_center - bounding_boxes.tensor[:, :2]

            center_manhatan: Tensor = torch.abs(bbs_center - center_bounding_box_center)
            center_manhatan_offset: Tensor = center_manhatan - bbs_half_size - center_bounding_box_half_size

            filter: Tensor = (center_manhatan_offset < max_distance).all(dim=1)
            bbs_to_merge: Boxes = Boxes.cat([bounding_boxes[filter], center_bounding_box])
            bounding_boxes = bounding_boxes[~filter]
            center_bounding_box = Boxes(torch.tensor(
                [
                    [
                        torch.min(bbs_to_merge.tensor[:, 0]),
                        torch.min(bbs_to_merge.tensor[:, 1]),
                        torch.max(bbs_to_merge.tensor[:, 2]),
                        torch.max(bbs_to_merge.tensor[:, 3]),
                    ]
                ]
            ))

        return center_bounding_box, bounding_boxes

    def __first_input(self, model_input: Tensor) -> None:

        print("Warming up masker model (yolov7-mask)...")

        self.__bbox_width: int = model_input.shape[-1] // 4
        self.__bbox_height: int = model_input.shape[-2] // 4
        self.__bbox_pix: int = self.__bbox_width * self.__bbox_height * 5

        if self.__optimize_model:

            true_device: nebullvm.tools.base.DeviceType = nebullvm.tools.base.DeviceType.GPU if self.__device.type == "cuda" else nebullvm.tools.base.DeviceType.CPU  # type: ignore

            root_folder: str = os.path.abspath("./optimized_models/masker")
            os.makedirs(root_folder, exist_ok=True)
            available_optimized_models = os.listdir(root_folder)
            print("Trying to load last optimized model (yolov7-mask)...")

            try:
                model_loaded: bool = False
                for optimized_model in available_optimized_models:
                    optimized_model_path: str = os.path.join(root_folder, optimized_model)
                    if not os.path.isdir(optimized_model_path):
                        continue
                    print(f"Trying to load optimized model -> {optimized_model}...")
                    try:
                        preload_model: speedster.BaseInferenceLearner = speedster.load_model(optimized_model_path)  # type: ignore
                    except:
                        print(f"Current folder is not an optimized model ({optimized_model}), skipping...")
                        print("Error trace: \n\n")
                        traceback.print_exc()
                        continue
                    if preload_model.network_parameters.input_infos[0].size == list(model_input.shape) and preload_model.device.type == true_device:
                        print("Last compatible optimized model found (yolov7-mask), testing...")
                        with torch.inference_mode():
                            preload_model(model_input)
                            preload_model(model_input)
                            preload_model(model_input)

                        model_loaded = True
                        self.__model = preload_model
                        break
                    else:
                        print(
                            f"Optimized model ({optimized_model}) is incompatible (input: [{list(model_input.shape)}, {preload_model.device.type}], model_input: [{preload_model.network_parameters.input_infos[0].size}, {true_device}), skipping...")
                        continue

                assert model_loaded, "Last compatible optimized model not found (yolov7-mask)"
                print("Last optimized model successfully loaded (yolov7-mask)...")

            except:
                print("Optimizing model (yolov7-mask)...")
                # optimize model
                # best optimization for 3080Ti TensorRT fp16 (half), balance accuracy and memory usage with better fps
                # [Speedster results on NVIDIA GeForce RTX 3080 Ti]
                # Metric       Original Model    Optimized Model    Improvement
                # -----------  ----------------  -----------------  -------------
                # backend      PYTORCH           TensorRT
                # latency      0.0206 sec/batch  0.0024 sec/batch   8.66x
                # throughput   48.51 data/sec    419.93 data/sec    8.66x
                # model size   91.27 MB          93.96 MB           0%
                # metric drop                    0.0043
                # techniques                     fp16

                # input data
                rand_input_data: list[Tuple[Tensor, Tuple[Tensor, ...]]] = [((torch.rand_like(model_input), ), torch.zeros(1)) for _ in range(100)]  # type: ignore
                preoptimize_model: speedster.BaseInferenceLearner = speedster.optimize_model(self.__model,  # type: ignore
                                                                                             rand_input_data,
                                                                                             store_latencies=True,
                                                                                             metric_drop_ths=10e-2,
                                                                                             optimization_time="unconstrained",
                                                                                             device=str(true_device))

                with torch.inference_mode():
                    preoptimize_model(model_input)
                    preoptimize_model(model_input)
                    preoptimize_model(model_input)

                model_path: str = os.path.join(root_folder, f"model_{len(available_optimized_models)}")
                speedster.save_model(preoptimize_model, model_path)

                self.__model = preoptimize_model
                print("Model optimized and saved (yolov7-mask)...")
                del rand_input_data
        else:
            print("Optimizing model disabled, skipping optimization (yolov7-mask)...")

        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary())
        self.__model(model_input)
        self.__model(model_input)
        self.__model(model_input)
        self.__is_traced = True
        print("Masker model (yolov7-mask) warmed up")

    def forward_once_maskonly(self, image: FrameRGB) -> MaskOnlyData:
        """Generate detected people mask from image provided"""

        image, original_size, top_pad, bottom_pad, left_pad, right_pad = Utilities.letterbox(
            image,
            self.__target_size,
            self.__letterbox_color,
            self.__letterbox_stride,
            self.__letterbox_box_mode
        )

        tensor_input: Tensor = torch.as_tensor(image, device=self.__device, dtype=self.__dtype)

        tensor_input = tensor_input.permute(2, 0, 1)

        if self.__flip_bgr_rgb:
            tensor_input = tensor_input[None, [2, 1, 0]]
        else:
            tensor_input = tensor_input[None, ...]

        tensor_input = tensor_input/255

        if not self.__is_traced:
            self.__first_input(tensor_input)

        with torch.inference_mode():
            yolo_output: Tensor = self.__model(tensor_input)

        if self.__optimize_model:
            yolo_output = yolo_output[0]

        inference_output, attenuation, bases = MaskGenerator.__unpack_inference(
            yolo_output,
            self.__bbox_width,
            self.__bbox_height,
            self.__bbox_pix
        )

        _, _, h, w = tensor_input.shape

        found_s, output_s, output_mask_s = MaskGenerator.__nms_conf_people_only_once(
            inference_output,
            attenuation,
            bases,
            self.__hyperparameters["attn_resolution"],
            self.__hyperparameters["num_base"],
            self.__pooler,  # type: ignore
            self.__confidence_threshold,
            self.__iou_threshold
        )

        human_detected: IsHumanDetected = False
        total_mask_np: SegmentationMask = numpy.zeros(image.shape[:-1], dtype=bool)

        if found_s:
            pred: Tensor = output_s
            pred_masks: Tensor = output_mask_s

            bboxes = Boxes(pred[:, :4])
            ori_pred_masks: Tensor = pred_masks.view(-1, self.__hyperparameters['mask_resolution'], self.__hyperparameters['mask_resolution'])
            pred_masks = paste_masks_in_image(ori_pred_masks, bboxes, (h, w), threshold=0.5)  # type:ignore

            # pytorch doesn't have a bitwise_or.reduce, so any in dimension 0 is used instead, improves time by half
            total_mask: Tensor = pred_masks.any(dim=0)
            total_mask_np = total_mask.to("cpu").numpy()

            human_detected = True

        total_mask_np = Utilities.unletterbox(
            total_mask_np,
            original_size,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
            self.__resize_result_to_original
        )

        return (human_detected, total_mask_np)

    def forward_once_with_mcbb(self, image: FrameRGB) -> MaskWithMostCenterBoundingBoxData:
        """Generate detected people mask from letterboxed image provided with most center bounding box"""

        image, original_size, top_pad, bottom_pad, left_pad, right_pad = Utilities.letterbox(
            image,
            self.__target_size,
            self.__letterbox_color,
            self.__letterbox_stride,
            self.__letterbox_box_mode
        )
        letterbox_size = image.shape[:-1]

        tensor_input: Tensor = torch.as_tensor(image, device=self.__device, dtype=self.__dtype)

        tensor_input = tensor_input.permute(2, 0, 1)

        if self.__flip_bgr_rgb:
            tensor_input = tensor_input[None, [2, 1, 0]]
        else:
            tensor_input = tensor_input[None, ...]

        tensor_input = tensor_input/255

        if not self.__is_traced:
            self.__first_input(tensor_input)

        with torch.inference_mode():
            yolo_output: Tensor = self.__model(tensor_input)

        if self.__optimize_model:
            yolo_output = yolo_output[0]

        inference_output, attenuation, bases = MaskGenerator.__unpack_inference(
            yolo_output,
            self.__bbox_width,
            self.__bbox_height,
            self.__bbox_pix
        )

        _, _, h, w = tensor_input.shape

        found_s, output_s, output_mask_s = MaskGenerator.__nms_conf_people_only_once(
            inference_output,
            attenuation,
            bases,
            self.__hyperparameters["attn_resolution"],
            self.__hyperparameters["num_base"],
            self.__pooler,  # type: ignore
            self.__confidence_threshold,
            self.__iou_threshold
        )

        human_detected: IsHumanDetected = False
        total_mask_np: SegmentationMask = numpy.zeros(image.shape[:-1], dtype=bool)
        most_center_bounding_box_np: BoundingBoxXY1XY2 = numpy.array((0, 0, image.shape[1], image.shape[0]), dtype=numpy.int16)

        if found_s:
            pred: Tensor = output_s
            pred_masks: Tensor = output_mask_s

            bboxes = Boxes(pred[:, :4])
            ori_pred_masks: Tensor = pred_masks.view(-1, self.__hyperparameters['mask_resolution'], self.__hyperparameters['mask_resolution'])
            pred_masks = paste_masks_in_image(ori_pred_masks, bboxes, (h, w), threshold=0.5)  # type:ignore

            # pytorch doesn't have a bitwise_or.reduce, so any in dimension 0 is used instead, improves time by half
            total_mask: Tensor = pred_masks.any(dim=0)
            total_mask_np = total_mask.to("cpu").numpy()

            most_center_bounding_box, bboxes = MaskGenerator.__find_most_center_bb(bboxes, (h, w), self.__bounding_box_grouping_range_scale, self.__bounding_box_no_merge)
            most_center_bounding_box_tensor: Tensor = torch.clamp_min(most_center_bounding_box.tensor, 0)
            most_center_bounding_box_np = numpy.around((most_center_bounding_box_tensor.to("cpu").numpy()[0].astype(numpy.uint16)))

            human_detected = True

        total_mask_np = Utilities.unletterbox(
            total_mask_np,
            original_size,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
            self.__resize_result_to_original
        )

        most_center_bounding_box_np = Utilities.unletterbox_bounding_box(
            most_center_bounding_box_np,
            letterbox_size,
            original_size,
            top_pad, bottom_pad, left_pad, right_pad,
            self.__resize_result_to_original
        )

        return (human_detected, total_mask_np, most_center_bounding_box_np)


class MaskGeneratorMocker(MaskGenerator):
    """Writes and reads generated mask and most center bounding box (MCBB) to and from a file, mocking MaskGenerator behaviour"""

    MASK_PATH = "masks"
    MASK_MCBB_PATH = "masks_mcbb"
    MASK_EXIST_PATH = "masks_exist"
    MASK_HWC_ATTR = "mask_hwc"
    MASK_COUNT_ATTR = "mask_count"

    def __init__(self, h5py_instance: Union[h5py.File, h5py.Group], *args, **kwargs) -> None:
        self.__mask: deque[SegmentationMask] = deque()
        self.__mask_mcbb: deque[BoundingBoxXY1XY2] = deque()
        self.__mask_exist: deque[IsHumanDetected] = deque()
        self.__h5py_instance: Union[h5py.File, h5py.Group] = h5py_instance

    def __len__(self):
        return len(self.__mask)

    def load(self, start_index: int = -1, stop_index: int = -1) -> bool:
        """Loads masks and most center bounding box (MCBB) from file to queues"""
        self.flush()

        if not all([x in self.__h5py_instance for x in [self.MASK_PATH, self.MASK_MCBB_PATH, self.MASK_EXIST_PATH]]):
            return False

        if start_index < 0:
            start_index = 0

        if stop_index < 0:
            stop_index = len(self.__h5py_instance[self.MASK_PATH])  # type: ignore

        self.__mask = deque(self.__h5py_instance[self.MASK_PATH][start_index:stop_index])  # type: ignore
        self.__mask_mcbb = deque(self.__h5py_instance[self.MASK_MCBB_PATH][start_index:stop_index])  # type: ignore
        self.__mask_exist = deque(self.__h5py_instance[self.MASK_EXIST_PATH][start_index:stop_index])  # type: ignore

        if len(self.__mask) != len(self.__mask_mcbb) != len(self.__mask_exist):
            self.flush()
            return False

        return True

    def save(self) -> bool:
        """Saves masks and most center bounding box (MCBB) in queues to file and flushes the queue"""
        if self.MASK_PATH in self.__h5py_instance:
            del self.__h5py_instance[self.MASK_PATH]
        if self.MASK_MCBB_PATH in self.__h5py_instance:
            del self.__h5py_instance[self.MASK_MCBB_PATH]
        if self.MASK_EXIST_PATH in self.__h5py_instance:
            del self.__h5py_instance[self.MASK_EXIST_PATH]

        self.__h5py_instance.create_dataset(name=self.MASK_PATH, data=numpy.array(list(self.__mask)),
                                            compression="gzip",
                                            )
        self.__h5py_instance.create_dataset(name=self.MASK_MCBB_PATH, data=numpy.array(list(self.__mask_mcbb)),
                                            compression="gzip",
                                            )
        self.__h5py_instance.create_dataset(name=self.MASK_EXIST_PATH, data=numpy.array(list(self.__mask_exist)),
                                            compression="gzip",
                                            )

        if not all(
            numpy.allclose(a, self.__h5py_instance[b][()]) for a, b in [  # type: ignore
                (self.__mask, self.MASK_PATH),
                (self.__mask_mcbb, self.MASK_MCBB_PATH),
                (self.__mask_exist, self.MASK_EXIST_PATH)
            ]
        ):
            return False

        self.__h5py_instance.attrs[self.MASK_HWC_ATTR] = self.__mask[0].shape
        self.__h5py_instance.attrs[self.MASK_COUNT_ATTR] = len(self.__mask)

        self.flush()
        return True

    def append(self, data: MaskWithMostCenterBoundingBoxData) -> None:
        """Appends masks and most center bounding box (MCBB) to respective queues"""
        self.__mask_exist.append(data[0])
        self.__mask.append(data[1].copy())
        self.__mask_mcbb.append(data[2].copy())

    def flush(self) -> None:
        """Clear all queues"""
        self.__mask.clear()
        self.__mask_mcbb.clear()
        self.__mask_exist.clear()

    def forward_once_maskonly(self, *args, **kwargs) -> MaskOnlyData:
        """Pops mask from queue simulates MaskGenerator's forward_once_maskonly"""
        if len(self.__mask) <= 0:
            return False, numpy.empty((0))
        mask: SegmentationMask = self.__mask.popleft()
        mask_exist: IsHumanDetected = self.__mask_exist.popleft()
        _ = self.__mask_mcbb.popleft()
        return mask_exist, mask

    def forward_once_with_mcbb(self, *args, **kwargs) -> MaskWithMostCenterBoundingBoxData:
        """Pops mask and most center bounding box (MCBB) from queue simulates MaskGenerator's forward_once_maskonly"""
        if len(self.__mask) <= 0:
            return False, numpy.zeros(0), numpy.zeros(0)
        mask: SegmentationMask = self.__mask.popleft()
        mask_mcbb: BoundingBoxXY1XY2 = self.__mask_mcbb.popleft()
        mask_exist: IsHumanDetected = self.__mask_exist.popleft()
        return mask_exist, mask, mask_mcbb
