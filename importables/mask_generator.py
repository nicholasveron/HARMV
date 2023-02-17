# pyright: reportPrivateImportUsage=false
"""Mask Generator generates peoples mask using YOLOv7 - Mask from bgr frames"""

import torch
import numba
import torchvision
from torchvision import transforms
import yaml
import numpy

from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.layers import paste_masks_in_image
import detectron2

from detectron2.structures import Boxes
from numpy import ndarray
from detectron2.modeling.poolers import ROIPooler
from torch import Tensor
import cv2


@numba.njit(fastmath=True)
def pre_rev_letterbox(
    shape: tuple[int, int],
    new_shape: int,
    stride: int
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    # Scale ratio (new / old)
    r = max(shape)
    r = new_shape / r

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]  # wh padding
    dw, dh = dw % stride, dh % stride  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    return new_unpad, (top, bottom, left, right)


def rev_letterbox(
        img: ndarray,
        new_shape: int = 640,
        color: tuple[int, int, int] = (114, 114, 114),
        stride: int = 32
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]

    new_unpad, (top, bottom, left, right) = pre_rev_letterbox(
        shape, new_shape, stride
    )

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img


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
    xi: int,
    attn: Tensor,
    bases: Tensor,
    attn_res: int,
    num_base: int,
    pooler: ROIPooler,
    conf_thres: float,
    iou_thres: float,
    multi_label: bool,
    max_wh: int,
    max_det: int
) -> tuple[bool, Tensor, Tensor]:
    output: Tensor = torch.empty((0))
    output_mask: Tensor = torch.empty((0))

    x = x[xc[xi]]  # confidence

    # If none remain process next image
    if not x.shape[0]:
        return False, output, output_mask

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box: Tensor = rev_xywh2xyxy(x[:, :4])

    a: Tensor = attn[xi][xc[xi]]
    base: Tensor = bases[xi]

    bboxes: Boxes = Boxes(box)

    base_li: list[Tensor] = [base[None]]
    bbox_li: list[Boxes] = [bboxes]
    pooled_bases: Tensor = pooler.forward(base_li, bbox_li)

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

    filter: Tensor = x[:, 5:6] == 0
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
                                      xc,
                                      xi,
                                      attn,
                                      bases,
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


MaskData = list[tuple[bool, ndarray, ndarray]]


class MaskGenerator:
    """Mask Generator generates peoples mask using YOLOv7 - Mask from bgr frames"""

    def __init__(self,
                 weight_path: str,
                 hyperparameter_path: str,
                 confidence_threshold: float,
                 iou_threshold: float
                 ) -> None:

        # check device capability
        self.__device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__half_capable: bool = self.__device.type != "cpu"

        # load model
        self.__model = torch.load(weight_path)['model'].to(self.__device)

        # set model precision
        if self.__half_capable:
            self.__model = self.__model.half()
        else:
            self.__model = self.__model.float()

        # set model to eval mode
        self.__model.eval()

        # load hyperparameters
        with open(hyperparameter_path) as hyp_file:  # pylint: disable=unspecified-encoding
            self.__hyperparameters: dict = yaml.load(hyp_file, Loader=yaml.FullLoader)

        # initialize pooler
        self.__pooler = torch.jit.script(
            ROIPooler(
                output_size=self.__hyperparameters['mask_resolution'],
                scales=(self.__model.pooler_scale,),
                sampling_ratio=1,
                pooler_type='ROIAlignV2',
                canonical_level=2)
        )

        # other parameters
        self.__confidence_threshold: float = confidence_threshold
        self.__iou_threshold: float = iou_threshold

    def generate(self, letterboxed_image_list: list[ndarray]) -> MaskData:
        """Generate detected people mask from letterboxed image provided"""
        tensor_image_list: list[Tensor] = []
        for image in letterboxed_image_list:
            tensor_image_list.append(transforms.ToTensor()(image).numpy())

        tensor_input: Tensor = torch.tensor(numpy.array(tensor_image_list), device=self.__device)
        tensor_input: Tensor = tensor_input.half() if self.__half_capable else tensor_input.float()

        with torch.no_grad():
            yolo_output: dict = self.__model(tensor_input)

        inference_output: Tensor = yolo_output['test']
        attenuation: Tensor = yolo_output['attn']
        bases: Tensor = torch.cat([yolo_output['bases'], yolo_output['sem']], dim=1)

        n, _, h, w = tensor_input.shape

        found, output, output_mask = rev_nms_conf_people_only(
            inference_output,
            attenuation,
            bases,
            self.__hyperparameters["attn_resolution"],
            self.__hyperparameters["num_base"],
            self.__pooler,  # type: ignore
            conf_thres=self.__confidence_threshold,
            iou_thres=self.__iou_threshold)

        results: list[tuple[bool, ndarray, ndarray]] = [
            (False, numpy.zeros((h, w), dtype=bool), numpy.array([], dtype=numpy.int64))
        ] * n

        for i in range(n):
            if found[i]:
                pred: Tensor = output[i]
                pred_masks = output_mask[i]

                bboxes = Boxes(pred[:, :4])
                ori_pred_masks = pred_masks.view(-1, self.__hyperparameters['mask_resolution'], self.__hyperparameters['mask_resolution'])
                pred_masks = retry_if_cuda_oom(paste_masks_in_image)(ori_pred_masks, bboxes, (h, w), threshold=0.5)

                # pytorch doesn't have a bitwise_or.reduce, so transpose and any is used instead, improves time by half
                pred_masks = torch.transpose(pred_masks, 1, 0)
                total_mask = torch.any(pred_masks, dim=1)
                total_mask_np = total_mask.detach().cpu().numpy()
                bboxes_np = bboxes.tensor.detach().cpu().numpy()

                results[i] = (True, total_mask_np, bboxes_np)

        return results
