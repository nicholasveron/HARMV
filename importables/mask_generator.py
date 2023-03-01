# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
"""Mask Generator generates peoples mask using YOLOv7 - Mask from bgr frames"""

import torch
import torchvision
import yaml
import numpy

from detectron2.layers import paste_masks_in_image

from detectron2.structures import Boxes
from numpy import ndarray
from detectron2.modeling.poolers import ROIPooler
from torch import Tensor
import h5py
from collections import deque


@torch.jit.script
def unpack_inference(x: Tensor, bbox_w: int, bbox_h: int, bbox_pix: int) -> tuple[Tensor, Tensor, Tensor]:
    inf: Tensor = x[:, :, :85]
    attn: Tensor = x[:, :, 85:1065]
    bases: Tensor = x[:, :, 1065:]

    bases = bases.flatten()
    bases = bases[:bbox_pix]
    bases = bases.reshape(1, 5, bbox_h, bbox_w)

    return inf, attn, bases


@torch.jit.script
def rev_xywh2xyxy(x: Tensor) -> Tensor:
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y: Tensor = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


@torch.jit.script
def rev_merge_bases(rois: Tensor, coeffs: Tensor, attn_r: int, num_b: int) -> Tensor:
    # merge predictions
    N, _, H, W = rois.size()
    if coeffs.dim() != 4:
        coeffs = coeffs.view(N, num_b, attn_r, attn_r)
    coeffs = torch.nn.functional.interpolate(coeffs, (H, W),
                                             mode="bilinear").softmax(dim=1)
    masks_preds: Tensor = (rois * coeffs).sum(dim=1)
    return masks_preds


def rev_loopable_nms_conf_people_only(
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
) -> tuple[bool, Tensor, Tensor]:
    output: Tensor = torch.empty((0), device=x.device, dtype=x.dtype)
    output_mask: Tensor = torch.empty((0), device=x.device, dtype=x.dtype)

    x = x[xc]  # confidence

    # If none remain process next image
    if not x.shape[0]:
        return False, output, output_mask

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box: Tensor = rev_xywh2xyxy(x[:, :4])

    a: Tensor = attn[xc]

    bboxes: Boxes = Boxes(box)

    base_li: list[Tensor] = [base[None]]
    bbox_li: list[Boxes] = [bboxes]
    pooled_bases: Tensor = pooler(base_li, bbox_li)

    pred_masks: Tensor = rev_merge_bases(pooled_bases, a, attn_res, num_base).view(a.shape[0], -1).sigmoid()

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


def rev_nms_conf_people_only(
    prediction: Tensor,
    attn: Tensor,
    bases: Tensor,
    attn_res: int,
    num_base: int,
    pooler: ROIPooler,
    conf_thres: float = 0.1,
    iou_thres: float = 0.6
) -> tuple[list[bool], list[Tensor], list[Tensor]]:
    nc: int = prediction[0].shape[1] - 5  # number of classes
    xc: Tensor = prediction[..., 4] > conf_thres  # candidates
    # Settings
    max_wh: int = 4096  # (pixels) minimum and maximum box width and height
    max_det: int = 300  # maximum number of detections per image
    multi_label: bool = nc > 1  # multiple labels per box (adds 0.5ms/img)

    found: list[bool] = []
    output: list[Tensor] = []
    output_mask: list[Tensor] = []

    futures: list[torch.jit.Future[tuple[bool, Tensor, Tensor]]] = []
    for xi in range(len(prediction)):  # image index, image inference
        futures.append(torch.jit.fork(rev_loopable_nms_conf_people_only,
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
                                      ))

    for future in futures:
        found_s, output_s, output_mask_s = torch.jit.wait(future)
        found.append(found_s)
        output.append(output_s)
        output_mask.append(output_mask_s)

    return found, output, output_mask


def rev_nms_conf_people_only_once(
    prediction: Tensor,
    attn: Tensor,
    bases: Tensor,
    attn_res: int,
    num_base: int,
    pooler: ROIPooler,
    conf_thres: float = 0.1,
    iou_thres: float = 0.6
) -> tuple[bool, Tensor, Tensor]:
    nc: int = prediction[0].shape[1] - 5  # number of classes
    xc: Tensor = prediction[..., 4] > conf_thres  # candidates
    # Settings
    max_wh: int = 4096  # (pixels) minimum and maximum box width and height
    max_det: int = 300  # maximum number of detections per image
    multi_label: bool = nc > 1  # multiple labels per box (adds 0.5ms/img)

    found_s, output_s, output_mask_s = rev_loopable_nms_conf_people_only(
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


MaskOnlyData = tuple[bool, ndarray]


class MaskGenerator:
    """Mask Generator generates peoples mask using YOLOv7 - Mask from bgr frames"""

    def __init__(self,
                 weight_path: str,
                 hyperparameter_path: str,
                 confidence_threshold: float,
                 iou_threshold: float
                 ) -> None:

        print("Initializing mask generator...")

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
        self.__model = self.__model.half() if self.__half_capable else self.__model.float()

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

        print("Mask generator initialized")

    def __first_input(self, model_input: Tensor) -> None:

        print("Warming up masker model (yolov7-mask)...")
        # trace model on first input for adaptive tracing

        self.__bbox_width: int = model_input.shape[-1] // 4
        self.__bbox_height: int = model_input.shape[-2] // 4
        self.__bbox_pix: int = self.__bbox_width * self.__bbox_height * 5
        self.__target_size: int = int(((model_input.shape[-2]*model_input.shape[-1])/25600)*1575)
        self.__offset_size: int = int(((self.__bbox_width * self.__bbox_height) / 1600) * 125)
        self.__pad_size: int = self.__target_size - (self.__offset_size % self.__target_size)
        self.__model.model[-1].pad_size = self.__pad_size  # type: ignore

        zero_input = torch.zeros_like(model_input).to(self.__device)
        rand_input = torch.rand_like(model_input).to(self.__device)
        rand_2_input = torch.rand_like(model_input).to(self.__device)

        with torch.no_grad():
            print("JIT tracing masker model (yolov7-mask)...")
            self.__model = torch.jit.trace(
                self.__model,
                rand_input,
                check_inputs=[
                    model_input,
                    zero_input,
                    rand_input,
                    rand_2_input
                ]
            )

            print("Optimizing masker model (yolov7-mask)...")
            self.__model = torch.jit.optimize_for_inference(self.__model)

        del zero_input
        del rand_input
        del rand_2_input
        torch.cuda.empty_cache()
        print(torch.cuda.memory_summary())
        self.__is_traced = True
        print("Masker model (yolov7-mask) warmed up")

    def forward_maskonly(self, letterboxed_image_list: list[ndarray], flip_bgr_rgb: bool = True) -> list[MaskOnlyData]:
        """Generate detected people mask from letterboxed image provided"""

        tensor_input: Tensor = torch.as_tensor(numpy.array(letterboxed_image_list), device=self.__device)
        tensor_input = tensor_input.permute(0, 3, 1, 2)

        if flip_bgr_rgb:
            tensor_input = tensor_input[:, [2, 1, 0]]

        tensor_input = tensor_input/255

        # only use full float because half causes some problems
        # tensor_input = tensor_input.half() if self.__half_capable else tensor_input.float()

        if not self.__is_traced:
            self.__first_input(tensor_input)

        with torch.no_grad():
            yolo_output = self.__model.forward(tensor_input).detach()  # type:ignore

        inference_output, attenuation, bases = unpack_inference(
            yolo_output,
            self.__bbox_width,
            self.__bbox_height,
            self.__bbox_pix
        )

        n, _, h, w = tensor_input.shape

        found, output, output_mask = rev_nms_conf_people_only(
            inference_output,
            attenuation,
            bases,
            self.__hyperparameters["attn_resolution"],
            self.__hyperparameters["num_base"],
            self.__pooler,  # type: ignore
            self.__confidence_threshold,
            self.__iou_threshold
        )

        results: list[MaskOnlyData] = [
            (False, numpy.zeros((h, w), dtype=bool))
        ] * n

        for i in range(n):
            if found[i]:
                pred: Tensor = output[i]
                pred_masks: Tensor = output_mask[i]

                bboxes = Boxes(pred[:, : 4])
                ori_pred_masks: Tensor = pred_masks.view(-1, self.__hyperparameters['mask_resolution'], self.__hyperparameters['mask_resolution'])
                pred_masks = paste_masks_in_image(ori_pred_masks, bboxes, (h, w), threshold=0.5)  # type:ignore

                # pytorch doesn't have a bitwise_or.reduce, so any in dimension 0 is used instead, improves time by half
                total_mask: Tensor = torch.any(pred_masks, dim=0)
                total_mask_np: ndarray = total_mask.to("cpu").numpy()

                results[i] = (True, total_mask_np)

        return results

    def forward_once_maskonly(self, letterboxed_image: ndarray, flip_bgr_rgb: bool = True) -> MaskOnlyData:
        """Generate detected people mask from letterboxed image provided"""
        tensor_input: Tensor = torch.as_tensor(letterboxed_image, device=self.__device)

        tensor_input = tensor_input.permute(2, 0, 1)

        if flip_bgr_rgb:
            tensor_input = tensor_input[None, [2, 1, 0]]
        else:
            tensor_input = tensor_input[None, ...]

        tensor_input = tensor_input/255

        # only use full float because half causes some problems
        # tensor_input = tensor_input.half() if self.__half_capable else tensor_input.float()

        if not self.__is_traced:
            self.__first_input(tensor_input)

        with torch.no_grad():
            yolo_output = self.__model.forward(tensor_input).detach()  # type:ignore

        inference_output, attenuation, bases = unpack_inference(
            yolo_output,
            self.__bbox_width,
            self.__bbox_height,
            self.__bbox_pix
        )

        _, _, h, w = tensor_input.shape

        found_s, output_s, output_mask_s = rev_nms_conf_people_only_once(
            inference_output,
            attenuation,
            bases,
            self.__hyperparameters["attn_resolution"],
            self.__hyperparameters["num_base"],
            self.__pooler,  # type: ignore
            self.__confidence_threshold,
            self.__iou_threshold
        )

        result: MaskOnlyData = False, numpy.zeros((h, w), dtype=bool)

        if found_s:
            pred: Tensor = output_s
            pred_masks: Tensor = output_mask_s

            bboxes = Boxes(pred[:, :4])
            ori_pred_masks: Tensor = pred_masks.view(-1, self.__hyperparameters['mask_resolution'], self.__hyperparameters['mask_resolution'])
            pred_masks = paste_masks_in_image(ori_pred_masks, bboxes, (h, w), threshold=0.5)  # type:ignore

            # pytorch doesn't have a bitwise_or.reduce, so any in dimension 0 is used instead, improves time by half
            total_mask: Tensor = pred_masks.any(dim=0)
            total_mask_np: ndarray = total_mask.to("cpu").numpy()

            result = (True, total_mask_np)

        return result


class MaskMocker(MaskGenerator):
    """Writes and reads generated mask to and from a file, mocking MaskGenerator behaviour"""

    MASK_PATH = "masks"
    MASK_EXIST_PATH = "masks_exist"
    MASK_HWC_ATTR = "mask_hwc"
    MASK_COUNT_ATTR = "mask_count"

    def __init__(self, h5py_instance: h5py.File, *args, **kwargs) -> None:
        self.__mask: deque[ndarray] = deque()
        self.__mask_exist: deque[bool] = deque()
        self.__h5py_instance: h5py.File = h5py_instance

    def load(self) -> bool:
        """Loads masks from file to queues"""
        self.flush()

        if not all([x in self.__h5py_instance for x in [self.MASK_PATH, self.MASK_EXIST_PATH]]):
            return False

        for mask in self.__h5py_instance[self.MASK_PATH][()]:  # type: ignore
            self.__mask.append(mask)
        for mask_exist in self.__h5py_instance[self.MASK_EXIST_PATH][()]:  # type: ignore
            self.__mask_exist.append(mask_exist)

        if len(self.__mask) != len(self.__mask_exist):
            self.flush()
            return False

        return True

    def save(self) -> bool:
        """Saves masks in queues to file and flushes the queue"""
        if self.MASK_PATH in self.__h5py_instance:
            del self.__h5py_instance[self.MASK_PATH]
        if self.MASK_EXIST_PATH in self.__h5py_instance:
            del self.__h5py_instance[self.MASK_EXIST_PATH]

        self.__h5py_instance.create_dataset(name=self.MASK_PATH, data=numpy.array(list(self.__mask)),
                                            compression="gzip",
                                            )
        self.__h5py_instance.create_dataset(name=self.MASK_EXIST_PATH, data=numpy.array(list(self.__mask_exist)),
                                            compression="gzip",
                                            )

        if not all(
            numpy.allclose(a, self.__h5py_instance[b][()]) for a, b in [  # type: ignore
                (self.__mask, self.MASK_PATH),
                (self.__mask_exist, self.MASK_EXIST_PATH)
            ]
        ):
            return False

        self.__h5py_instance.attrs[self.MASK_HWC_ATTR] = self.__mask[0].shape
        self.__h5py_instance.attrs[self.MASK_COUNT_ATTR] = len(self.__mask)

        self.flush()
        return True

    def append(self, data: MaskOnlyData) -> None:
        """Appends masks to respective queues"""
        self.__mask_exist.append(data[0])
        self.__mask.append(data[1].copy())

    def flush(self) -> None:
        """Clear all queues"""
        self.__mask.clear()
        self.__mask_exist.clear()

    def forward_once_maskonly(self, *args, **kwargs) -> MaskOnlyData:
        """Pops mask from queue simulates MaskGenerator's forward_once_maskonly"""
        if len(self.__mask) <= 0:
            return False, numpy.empty((0))
        mask: ndarray = self.__mask.popleft()
        mask_exist: bool = self.__mask_exist.popleft()
        return mask_exist, mask

    def forward_maskonly(self, *args, **kwargs) -> list[MaskOnlyData]:
        """Pops all masks from queue simulates MaskGenerator's forward_maskonly"""
        masks: list[MaskOnlyData] = []
        while len(self.__mask) > 0:
            masks.append(self.forward_once_maskonly())
        return masks
