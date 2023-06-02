
# run rtsp
# https://github.com/aler9/rtsp-simple-server#from-a-webcam
# import the opencv library
import cv2
import h5py
import time
import numpy
import torch
import sklearn.preprocessing
import torchvision
from collections import deque
from copy import deepcopy
from ptlflow.utils import flow_utils
from importables.utilities import Utilities
from importables.video_decoder import VideoDecoderProcessSpawner
from importables.mask_generator import MaskGenerator, MaskGeneratorMocker
from importables.custom_datasets import FlowDataset
from importables.custom_types import (
    Union,
    Tensor,
    ndarray,
    FrameRGB,
    DecodedData,
    RawMotionVectors,
    SegmentationMask,
    BoundingBoxXY1XY2,
    MotionVectorFrame,
    MaskWithMostCenterBoundingBoxData,
)
from importables.motion_vector_processor import MotionVectorProcessor, MotionVectorProcessorMocker
from importables.constants import (
    DEFAULT_TRAINING_PARAMETERS,
    PREPROCESSING_ARGUMENTS_MASK_GENERATOR,
    PREPROCESSING_ARGUMENTS_MOTION_VECTOR_PROCESSOR,
    SELECTED_NTU_MUTUAL_ACTIONS_SET,
    SELECTED_NTU_DAILY_ACTIONS_SET,
    NTU_ACTION_DATASET_MAP,
)


webcam_ip = "rtsp://NicholasXPS17:8554/cam"
video_path = "./S001C003P001R001A050_rgb.mp4"

# ----------- NORMAL RUN
args_decoder = {
    "path":  video_path,
    "realtime": False,
    "update_rate":  30,
}

label_encoder = sklearn.preprocessing.LabelEncoder().fit(numpy.expand_dims(sorted(list(SELECTED_NTU_DAILY_ACTIONS_SET | SELECTED_NTU_MUTUAL_ACTIONS_SET)), 1))
print(label_encoder.classes_)

# args_decoder = {
#     "path":  webcam_ip,
#     "realtime": True,
#     "update_rate":  30,
# }

decoder = VideoDecoderProcessSpawner(**args_decoder).start()
frame_data: DecodedData = decoder.read()
assert frame_data[0], "DECODER ERROR, ABORTING..."
decoder.stop()
last_frame_data: DecodedData = deepcopy(frame_data)
rgb_frame: FrameRGB = frame_data[1]
raw_motion_vector: RawMotionVectors = frame_data[2]

PREPROCESSING_ARGUMENTS_MOTION_VECTOR_PROCESSOR["input_size"] = rgb_frame.shape[:2]

motion_vector_processor: MotionVectorProcessor = MotionVectorProcessor(**PREPROCESSING_ARGUMENTS_MOTION_VECTOR_PROCESSOR)
mask_generator: MaskGenerator = MaskGenerator(**PREPROCESSING_ARGUMENTS_MASK_GENERATOR)

motion_vector_processor.process(raw_motion_vector)
motion_vector_processor.process(raw_motion_vector)
motion_vector_processor.process(raw_motion_vector)
mask_generator.forward_once_with_mcbb(rgb_frame)
mask_generator.forward_once_with_mcbb(rgb_frame)
mask_generator.forward_once_with_mcbb(rgb_frame)


# target_mcbb: BoundingBoxXY1XY2 = numpy.array((0, 0, mv_frame.shape[1], mv_frame.shape[0])).astype(numpy.float16)

print("Loading model ...")
inference_model = torch.load(
    "./tensorboard_logs/5_model_experiments/5_1_lstm_output_vs_lstm_hidden_vs_lstm_hidden_and_cell/runs/1684660486 - motion_vector - HARMV_CNNLSTM_ShuffleNetv2x0_5_Single/model_checkpoints/7")
inference_model = inference_model.to("cuda").eval()
print("Warming up model ...")
inference_model(
    torch.zeros((1, DEFAULT_TRAINING_PARAMETERS["timesteps"], 2, inference_model.resolution(), inference_model.resolution())).to("cuda")
)
print("Model loaded and warmed up")

metric_average = 100
preprocesing_deq = deque(maxlen=metric_average)
inference_deq = deque(maxlen=metric_average)
total_deq = deque(maxlen=metric_average)

video_deq = deque(maxlen=DEFAULT_TRAINING_PARAMETERS["timesteps"])

single_frame_preprocess = torchvision.transforms.Compose([
    FlowDataset.CropMask(crop=DEFAULT_TRAINING_PARAMETERS["crop"], mask=DEFAULT_TRAINING_PARAMETERS["mask"], replace_with=(0, 0)),
    FlowDataset.Bound(DEFAULT_TRAINING_PARAMETERS["bounding_value"]),
    FlowDataset.PadResize(inference_model.resolution(), pad_with=(128, 128)),
])

batchwise_preprocess = torchvision.transforms.Compose([
    FlowDataset.Rescale(1/255.),
    FlowDataset.BatchToCHW()
])


def preprocess_frame(current_frame_returns: dict):
    current_frame_returns = single_frame_preprocess(current_frame_returns)
    video_deq.append(current_frame_returns)

    if len(video_deq) == DEFAULT_TRAINING_PARAMETERS["timesteps"]:
        if "segmentation_mask" in current_frame_returns:
            del current_frame_returns["segmentation_mask"]
            del current_frame_returns["bounding_box"]

        collated_timestep: dict[str, Tensor] = torch.utils.data.default_collate(list(video_deq))
        collated_timestep = batchwise_preprocess(collated_timestep)

        return collated_timestep


while (True):
    video_deq.clear()

    path = input("Path (n to stop): ").strip()
    if path == "n":
        break
    realtime = input("Realtime? (y/any for n): ").strip().lower() == "y"
    update_rate = int(input("Update rate (int): ").strip())

    args_decoder = {
        "path":  path,
        "realtime": realtime,
        "update_rate":  update_rate,
    }

    decoder = VideoDecoderProcessSpawner(**args_decoder).start()
    frame_data = decoder.read()
    assert frame_data[0], "DECODER ERROR, ABORTING..."
    last_frame_data = deepcopy(frame_data)

    while (True):

        if cv2.waitKey(1) & 0xFF == ord('q') or not frame_data[0]:
            decoder.stop()
            break

        start_time_all = time.perf_counter()
        frame_data = decoder.read()

        available, rgb_frame, raw_motion_vectors, _ = frame_data
        _, last_rgb_frame, _, _ = last_frame_data

        if not available:
            break

        motion_vector_frame: MotionVectorFrame = motion_vector_processor.process(raw_motion_vectors)
        mask_data: MaskWithMostCenterBoundingBoxData = mask_generator.forward_once_with_mcbb(rgb_frame)
        # last_frame_data = frame_data

        current_frame_returns: dict[str, Union[ndarray, Tensor]] = {}
        is_mask_available, segmentation_mask, bounding_box = mask_data
        if is_mask_available:
            current_frame_returns["segmentation_mask"] = segmentation_mask
            current_frame_returns["bounding_box"] = bounding_box
        current_frame_returns["motion_vector"] = motion_vector_frame

        model_input = preprocess_frame(current_frame_returns)
        preprocesing_deq.append(1/(time.perf_counter()-start_time_all))
        start_time_inference = time.perf_counter()

        if model_input:
            with torch.no_grad():
                model_output = inference_model(model_input["motion_vector"].to("cuda")[None, ...])
                model_output = model_output.softmax(dim=1)
                pred_index = torch.argmax(model_output, dim=1).cpu().numpy()
                pred_conf = model_output[0, pred_index].cpu().numpy()[0]
                pred_a_id = label_encoder.inverse_transform(pred_index[..., None])[0]
                print(NTU_ACTION_DATASET_MAP[pred_a_id], str(round(pred_conf*100, 2)) + "%",
                      f"Prep: {numpy.mean(list(preprocesing_deq))}",
                      f"Infe: {numpy.mean(list(inference_deq))}")
        else:
            print(f"Filling queue: ({len(video_deq)}/{DEFAULT_TRAINING_PARAMETERS['timesteps']})")

        inference_deq.append(1/(time.perf_counter()-start_time_inference))

        # rgb_frame: FrameRGB = frame_data[1]
        # raw_mv: RawMotionVectors = frame_data[2]
        # mv_frame: MotionVectorFrame = mv_processor.process(raw_mv)
        # mask_data: MaskWithMostCenterBoundingBoxData = yolo_maskgen.forward_once_with_mcbb(rgb_frame)
        # fr = cv2.resize(rgb_frame, (mv_frame.shape[1], mv_frame.shape[0]))

        # fl_x = mv_frame[:, :, 0]
        # fl_y = mv_frame[:, :, 1]

        # mcbb: BoundingBoxXY1XY2 = mask_data[2]
        # target_mcbb = (target_mcbb + (mcbb.astype(numpy.float32) - target_mcbb) * 0.1)
        # mask_frame: SegmentationMask = mask_data[1]

        # mask_frame_uint = (mask_frame.astype(numpy.uint8)*255)

        # combine_to_mask = numpy.dstack(
        #     (
        #         numpy.copy(fr),
        #         numpy.copy(fl_x),
        #         numpy.copy(fl_y)
        #     )
        # )

        # combine_to_mask[~mask_frame] = (255, 255, 255, 128, 128)

        # tl_avg = list(timer_avg)
        # tl_avg.sort()
        # tl_avg_10 = tl_avg[:len(timer_avg)//10]
        # tl_avg_1 = tl_avg[:len(timer_avg)//100]
        # print(sum(timer_avg)//len(timer_avg), sum(tl_avg_10)//len(tl_avg_10), sum(tl_avg_1)//len(tl_avg_1), end="\n\n")
        # print(sum(timer_avg)//len(timer_avg), sum(tl_avg_10)//len(tl_avg_10), sum(tl_avg_1)//len(tl_avg_1), end="\r")

        # fmasked = combine_to_mask[:, :, 0:3]
        # fl_x_masked = combine_to_mask[:, :, 3]
        # fl_y_masked = combine_to_mask[:, :, 4]

        # fl_x = numpy.dstack((fl_x, fl_x, fl_x))
        # fl_y = numpy.dstack((fl_y, fl_y, fl_y))

        # fl_x_s = numpy.dstack((fl_x_masked, fl_x_masked, fl_x_masked))
        # fl_y_s = numpy.dstack((fl_y_masked, fl_y_masked, fl_y_masked))

        # mf = numpy.dstack((mask_frame_uint, mask_frame_uint, mask_frame_uint))

        # fmasked_crop = Utilities.crop_to_bb_and_resize(fmasked, target_mcbb, fr.shape[:2], fr.shape[:2])
        # fl_x_s_crop = Utilities.crop_to_bb_and_resize(fl_x_s, target_mcbb, fr.shape[:2], fr.shape[:2])
        # fl_y_s_crop = Utilities.crop_to_bb_and_resize(fl_y_s, target_mcbb, fr.shape[:2], fr.shape[:2])

        # fr = cv2.rectangle(fr, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        # fl_x = cv2.rectangle(fl_x, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        # fl_y = cv2.rectangle(fl_y, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
        # mf = cv2.rectangle(mf, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

        # frshow = numpy.hstack((fr, mf, fmasked, fmasked_crop))
        # flxshow = numpy.hstack((fl_x, mf, fl_x_s, fl_x_s_crop))
        # flyshow = numpy.hstack((fl_y, mf, fl_y_s, fl_y_s_crop))

        # fshow = numpy.vstack(
        #     (
        #         frshow,
        #         flxshow,
        #         flyshow,
        #         # flxyrgb
        #     )
        # )

        cv2.imshow('frame', cv2.resize(rgb_frame, (rgb_frame.shape[1]//2, rgb_frame.shape[0]//2)))

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice

    # After the loop release the cap object
    # Destroy all the windows
    decoder.stop()
    cv2.destroyAllWindows()
