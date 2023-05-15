
# run rtsp
# https://github.com/aler9/rtsp-simple-server#from-a-webcam
# import the opencv library
import cv2
import h5py
import time
import numpy
from collections import deque
from ptlflow.utils import flow_utils
from importables.utilities import Utilities
from importables.video_decoder import VideoDecoderProcessSpawner
from importables.mask_generator import MaskGenerator, MaskGeneratorMocker
from importables.custom_types import (
    FrameRGB,
    DecodedData,
    RawMotionVectors,
    SegmentationMask,
    BoundingBoxXY1XY2,
    MotionVectorFrame,
    MaskWithMostCenterBoundingBoxData,
)
from importables.motion_vector_processor import MotionVectorProcessor, MotionVectorProcessorMocker

timer_avg = deque([0.]*101, maxlen=101)

webcam_ip = "rtsp://NicholasXPS17:8554/cam"
video_path = "/mnt/c/Skripsi/dataset-h264/R001A016/S001C001P001R001A016_rgb.mp4"

# ----------- NORMAL RUN
args_decoder = {
    "path":  video_path,
    "realtime": False,
    "update_rate":  25,
}

# args_decoder = {
#     "path":  webcam_ip,
#     "realtime": True,
#     "update_rate":  120,
# }

decoder = VideoDecoderProcessSpawner(**args_decoder).start()
frame_data: DecodedData = decoder.read()

assert frame_data[0], "DECODER ERROR, ABORTING..."

args_mvprocessor = {
    "input_size": frame_data[1].shape[:2],
    "bound":  32,
    "raw_motion_vectors": False,
    "target_size": 320,
}
mv_processor = MotionVectorProcessor(**args_mvprocessor)
mv_frame: MotionVectorFrame = mv_processor.process(frame_data[2])

args_yolo = {
    "weight_path": './libs/yolov7-mask/yolov7-mask.pt',
    "hyperparameter_path": './libs/yolov7-mask/data/hyp.scratch.mask.yaml',
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "target_size": 320,
    "optimize_model": True,
}
yolo_maskgen = MaskGenerator(**args_yolo)
yolo_maskgen.forward_once_maskonly(frame_data[1])

counter = 0
target_mcbb: BoundingBoxXY1XY2 = numpy.array((0, 0, mv_frame.shape[1], mv_frame.shape[0])).astype(numpy.float16)

while(True):
    counter += 1
    start_time = time.perf_counter()
    # Capture the video frame by frame

    if cv2.waitKey(1) & 0xFF == ord('q') or not frame_data[0]:
        decoder.stop()
        break

    rgb_frame: FrameRGB = frame_data[1]
    raw_mv: RawMotionVectors = frame_data[2]

    mv_frame: MotionVectorFrame = mv_processor.process(raw_mv)

    start_time = time.perf_counter()
    mask_data: MaskWithMostCenterBoundingBoxData = yolo_maskgen.forward_once_with_mcbb(rgb_frame)
    timer_avg.append(1/((time.perf_counter()-start_time)))

    fr = cv2.resize(rgb_frame, (mv_frame.shape[1], mv_frame.shape[0]))

    fl_x = mv_frame[:, :, 0]
    fl_y = mv_frame[:, :, 1]

    mcbb: BoundingBoxXY1XY2 = mask_data[2]
    target_mcbb = (target_mcbb + (mcbb.astype(numpy.float32) - target_mcbb) * 0.1)
    mask_frame: SegmentationMask = mask_data[1]

    mask_frame_uint = (mask_frame.astype(numpy.uint8)*255)

    combine_to_mask = numpy.dstack(
        (
            numpy.copy(fr),
            numpy.copy(fl_x),
            numpy.copy(fl_y)
        )
    )

    combine_to_mask[~mask_frame] = (255, 255, 255, 128, 128)

    tl_avg = list(timer_avg)
    tl_avg.sort()
    tl_avg_10 = tl_avg[:len(timer_avg)//10]
    tl_avg_1 = tl_avg[:len(timer_avg)//100]
    print(sum(timer_avg)//len(timer_avg), sum(tl_avg_10)//len(tl_avg_10), sum(tl_avg_1)//len(tl_avg_1), end="\n\n")
    print(sum(timer_avg)//len(timer_avg), sum(tl_avg_10)//len(tl_avg_10), sum(tl_avg_1)//len(tl_avg_1), end="\r")

    fmasked = combine_to_mask[:, :, 0:3]
    fl_x_masked = combine_to_mask[:, :, 3]
    fl_y_masked = combine_to_mask[:, :, 4]

    fl_x = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y = numpy.dstack((fl_y, fl_y, fl_y))

    fl_x_s = numpy.dstack((fl_x_masked, fl_x_masked, fl_x_masked))
    fl_y_s = numpy.dstack((fl_y_masked, fl_y_masked, fl_y_masked))

    mf = numpy.dstack((mask_frame_uint, mask_frame_uint, mask_frame_uint))

    fmasked_crop = Utilities.crop_to_bb_and_resize(fmasked, target_mcbb, fr.shape[:2], fr.shape[:2])
    fl_x_s_crop = Utilities.crop_to_bb_and_resize(fl_x_s, target_mcbb, fr.shape[:2], fr.shape[:2])
    fl_y_s_crop = Utilities.crop_to_bb_and_resize(fl_y_s, target_mcbb, fr.shape[:2], fr.shape[:2])

    fr = cv2.rectangle(fr, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    fl_x = cv2.rectangle(fl_x, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    fl_y = cv2.rectangle(fl_y, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    mf = cv2.rectangle(mf, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    frshow = numpy.hstack((fr, mf, fmasked, fmasked_crop))
    flxshow = numpy.hstack((fl_x, mf, fl_x_s, fl_x_s_crop))
    flyshow = numpy.hstack((fl_y, mf, fl_y_s, fl_y_s_crop))

    fshow = numpy.vstack(
        (
            frshow,
            flxshow,
            flyshow,
            # flxyrgb
        )
    )

    cv2.imshow('frame', fshow)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    frame_data = decoder.read()

# After the loop release the cap object
# Destroy all the windows
decoder.stop()
cv2.destroyAllWindows()


# ----------- SAVING
decoder = VideoDecoderProcessSpawner(**args_decoder).start()
frame_data: DecodedData = decoder.read()

assert frame_data[0], "DECODER ERROR, ABORTING..."

args_mvprocessor = {
    "input_size": frame_data[1].shape[:2],
    "bound":  32,
    "raw_motion_vectors": True,
}

mv_processor = MotionVectorProcessor(**args_mvprocessor)
mv_frame: MotionVectorFrame = mv_processor.process(frame_data[2])

mve_save = h5py.File("try_mve.h5", mode="w")
mock_mve = MotionVectorProcessorMocker(mve_save, **args_mvprocessor)
mock_yolo_maskgen = MaskGeneratorMocker(mve_save, **args_yolo)

counter = 0

target_mcbb: BoundingBoxXY1XY2 = numpy.array((0, 0, mv_frame.shape[1], mv_frame.shape[0])).astype(numpy.float16)

while(True):
    counter += 1
    start_time = time.perf_counter()
    # Capture the video frame by frame

    if cv2.waitKey(1) & 0xFF == ord('q') or not frame_data[0]:
        decoder.stop()
        break

    rgb_frame: FrameRGB = frame_data[1]
    raw_mv: RawMotionVectors = frame_data[2]

    mv_frame: MotionVectorFrame = mv_processor.process(raw_mv)
    mock_mve.append(mv_frame)
    mv_frame = Utilities.bound_motion_frame(mv_frame.copy(), 128, 255/(2*args_mvprocessor["bound"]))

    start_time = time.perf_counter()
    mask_data: MaskWithMostCenterBoundingBoxData = yolo_maskgen.forward_once_with_mcbb(rgb_frame)
    mock_yolo_maskgen.append(mask_data)
    timer_avg.append(1/((time.perf_counter()-start_time)))

    fr = cv2.resize(rgb_frame, (mv_frame.shape[1], mv_frame.shape[0]))

    fl_x = mv_frame[:, :, 0]
    fl_y = mv_frame[:, :, 1]

    mcbb: BoundingBoxXY1XY2 = mask_data[2]
    target_mcbb = (target_mcbb + (mcbb.astype(numpy.float32) - target_mcbb) * 0.1)
    mask_frame: SegmentationMask = mask_data[1]

    mask_frame_uint = (mask_frame.astype(numpy.uint8)*255)

    combine_to_mask = numpy.dstack(
        (
            numpy.copy(fr),
            numpy.copy(fl_x),
            numpy.copy(fl_y)
        )
    )

    combine_to_mask[~mask_frame] = (255, 255, 255, 128, 128)

    tl_avg = list(timer_avg)
    tl_avg.sort()
    tl_avg_10 = tl_avg[:len(timer_avg)//10]
    tl_avg_1 = tl_avg[:len(timer_avg)//100]
    print(sum(timer_avg)//len(timer_avg), sum(tl_avg_10)//len(tl_avg_10), sum(tl_avg_1)//len(tl_avg_1), end="\n\n")
    print(sum(timer_avg)//len(timer_avg), sum(tl_avg_10)//len(tl_avg_10), sum(tl_avg_1)//len(tl_avg_1), end="\r")

    fmasked = combine_to_mask[:, :, 0:3]
    fl_x_masked = combine_to_mask[:, :, 3]
    fl_y_masked = combine_to_mask[:, :, 4]

    fl_x = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y = numpy.dstack((fl_y, fl_y, fl_y))

    fl_x_s = numpy.dstack((fl_x_masked, fl_x_masked, fl_x_masked))
    fl_y_s = numpy.dstack((fl_y_masked, fl_y_masked, fl_y_masked))

    mf = numpy.dstack((mask_frame_uint, mask_frame_uint, mask_frame_uint))

    fmasked_crop = Utilities.crop_to_bb_and_resize(fmasked, target_mcbb, fr.shape[:2], fr.shape[:2])
    fl_x_s_crop = Utilities.crop_to_bb_and_resize(fl_x_s, target_mcbb, fr.shape[:2], fr.shape[:2])
    fl_y_s_crop = Utilities.crop_to_bb_and_resize(fl_y_s, target_mcbb, fr.shape[:2], fr.shape[:2])

    fr = cv2.rectangle(fr, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    fl_x = cv2.rectangle(fl_x, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    fl_y = cv2.rectangle(fl_y, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    mf = cv2.rectangle(mf, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    frshow = numpy.hstack((fr, mf, fmasked, fmasked_crop))
    flxshow = numpy.hstack((fl_x, mf, fl_x_s, fl_x_s_crop))
    flyshow = numpy.hstack((fl_y, mf, fl_y_s, fl_y_s_crop))

    fshow = numpy.vstack(
        (
            frshow,
            flxshow,
            flyshow,
            # flxyrgb
        )
    )

    cv2.imshow('frame', fshow)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    frame_data = decoder.read()

# After the loop release the cap object
# Destroy all the windows
decoder.stop()
cv2.destroyAllWindows()
mock_mve.save()
mock_yolo_maskgen.save()
mve_save.close()
print(counter, f"{round((101-counter)/101, 1)}% loss")

# 3 640 640
# Half(1, 25200, 980, strides=[24696000, 980, 1], requires_grad=0, device=cuda:0),
# Half(1, 5, 160, 160, strides=[128000, 25600, 160, 1], requires_grad=0, device=cuda:0),
# Half(1, 25200, 85, strides=[2142000, 85, 1], requires_grad=0, device=cuda:0)

# 3 320 320
# Half(1, 6300, 980, strides=[6174000, 980, 1], requires_grad=0, device=cuda:0),
# Half(1, 5, 80, 80, strides=[32000, 6400, 80, 1], requires_grad=0, device=cuda:0),
# Half(1, 6300, 85, strides=[535500, 85, 1], requires_grad=0, device=cuda:0)

# 3 192 320
# Half(1, 3780, 980, strides=[3704400, 980, 1], requires_grad=0, device=cuda:0),
# Half(1, 5, 48, 80, strides=[19200, 3840, 80, 1], requires_grad=0, device=cuda:0),
# Half(1, 3780, 85, strides=[321300, 85, 1], requires_grad=0, device=cuda:0)

# 3 384 640
# Half(1, 15120, 980, strides=[14817600, 980, 1], requires_grad=0, device=cuda:0),
# Half(1, 5, 96, 160, strides=[76800, 15360, 160, 1], requires_grad=0, device=cuda:0),
# Half(1, 15120, 85, strides=[1285200, 85, 1], requires_grad=0, device=cuda:0)

# shape define ->
# test & attn same resolution (count my scale relative to 160*160 -> 1575) , dimension stackable = 980 : 85
# bases resolution = (img.shape/4) * 5

# stackable plan
# calculate pix = (img.shape.w * img.shape.h)

# calculate ratio_res = (pix / 25600)
# calculate test & attn reso  (ratio_res * 1575)

# calculate base_reso (pix / 16)
# bases pad size = test & attn reso - (ratio_res * 125)

# ----------- LOADING
decoder = VideoDecoderProcessSpawner(**args_decoder).start()
frame_data: DecodedData = decoder.read()

assert frame_data[0], "DECODER ERROR, ABORTING..."

args_mvprocessor = {
    "input_size": frame_data[1].shape[:2],
    "bound":  32,
    "raw_motion_vectors": False,
}
mv_processor = MotionVectorProcessor(**args_mvprocessor)
mv_frame: MotionVectorFrame = mv_processor.process(frame_data[2])


mve_load = h5py.File("try_mve.h5", mode="r")
mv_processor_mock = MotionVectorProcessorMocker(mve_load, **args_mvprocessor)
mv_processor_mock.load()
mock_yolo_maskgen = MaskGeneratorMocker(mve_load, **args_yolo)
mock_yolo_maskgen.load()

counter = 0

target_mcbb: BoundingBoxXY1XY2 = numpy.array((0, 0, mv_frame.shape[1], mv_frame.shape[0])).astype(numpy.float16)

while(True):
    counter += 1
    start_time = time.perf_counter()
    # Capture the video frame by frame

    if cv2.waitKey(1) & 0xFF == ord('q') or not frame_data[0]:
        decoder.stop()
        break
    # real
    rgb_frame: FrameRGB = frame_data[1]
    raw_mv: RawMotionVectors = frame_data[2]

    mv_frame: MotionVectorFrame = mv_processor.process(raw_mv)

    start_time = time.perf_counter()
    mask_data: MaskWithMostCenterBoundingBoxData = yolo_maskgen.forward_once_with_mcbb(rgb_frame)
    timer_avg.append(1/((time.perf_counter()-start_time)))

    fr = cv2.resize(rgb_frame, (mv_frame.shape[1], mv_frame.shape[0]))

    fl_x = mv_frame[:, :, 0]
    fl_y = mv_frame[:, :, 1]

    mcbb: BoundingBoxXY1XY2 = mask_data[2]
    target_mcbb = (target_mcbb + (mcbb.astype(numpy.float32) - target_mcbb) * 0.1)
    mask_frame: SegmentationMask = mask_data[1]

    mask_frame_uint = (mask_frame.astype(numpy.uint8)*255)

    combine_to_mask = numpy.dstack(
        (
            numpy.copy(fr),
            numpy.copy(fl_x),
            numpy.copy(fl_y)
        )
    )

    combine_to_mask[~mask_frame] = (255, 255, 255, 128, 128)

    tl_avg = list(timer_avg)
    tl_avg.sort()
    tl_avg_10 = tl_avg[:len(timer_avg)//10]
    tl_avg_1 = tl_avg[:len(timer_avg)//100]
    print(sum(timer_avg)//len(timer_avg), sum(tl_avg_10)//len(tl_avg_10), sum(tl_avg_1)//len(tl_avg_1), end="\n\n")
    print(sum(timer_avg)//len(timer_avg), sum(tl_avg_10)//len(tl_avg_10), sum(tl_avg_1)//len(tl_avg_1), end="\r")

    fmasked = combine_to_mask[:, :, 0:3]
    fl_x_masked = combine_to_mask[:, :, 3]
    fl_y_masked = combine_to_mask[:, :, 4]

    fl_x = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y = numpy.dstack((fl_y, fl_y, fl_y))

    fl_x_s = numpy.dstack((fl_x_masked, fl_x_masked, fl_x_masked))
    fl_y_s = numpy.dstack((fl_y_masked, fl_y_masked, fl_y_masked))

    mf = numpy.dstack((mask_frame_uint, mask_frame_uint, mask_frame_uint))

    fmasked_crop = Utilities.crop_to_bb_and_resize(fmasked, target_mcbb, fr.shape[:2], fr.shape[:2])
    fl_x_s_crop = Utilities.crop_to_bb_and_resize(fl_x_s, target_mcbb, fr.shape[:2], fr.shape[:2])
    fl_y_s_crop = Utilities.crop_to_bb_and_resize(fl_y_s, target_mcbb, fr.shape[:2], fr.shape[:2])

    fr = cv2.rectangle(fr, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    fl_x = cv2.rectangle(fl_x, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    fl_y = cv2.rectangle(fl_y, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    mf = cv2.rectangle(mf, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    frshow = numpy.hstack((fr, mf, fmasked, fmasked_crop))
    flxshow = numpy.hstack((fl_x, mf, fl_x_s, fl_x_s_crop))
    flyshow = numpy.hstack((fl_y, mf, fl_y_s, fl_y_s_crop))

    fshow = numpy.vstack(
        (
            # frshow,
            flxshow,
            flyshow,
            # flxyrgb
        )
    )

    # mock
    mv_frame: MotionVectorFrame = mv_processor_mock.process(raw_mv)

    mask_data: MaskWithMostCenterBoundingBoxData = mock_yolo_maskgen.forward_once_with_mcbb(rgb_frame)

    fl_x = mv_frame[:, :, 0]
    fl_y = mv_frame[:, :, 1]

    mcbb: BoundingBoxXY1XY2 = mask_data[2]
    target_mcbb = (target_mcbb + (mcbb.astype(numpy.float32) - target_mcbb) * 0.1)
    mask_frame: SegmentationMask = mask_data[1]

    mask_frame_uint = (mask_frame.astype(numpy.uint8)*255)

    combine_to_mask = numpy.dstack(
        (
            numpy.copy(fr),
            numpy.copy(fl_x),
            numpy.copy(fl_y)
        )
    )

    combine_to_mask[~mask_frame] = (255, 255, 255, 128, 128)

    fmasked = combine_to_mask[:, :, 0:3]
    fl_x_masked = combine_to_mask[:, :, 3]
    fl_y_masked = combine_to_mask[:, :, 4]

    fl_x = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y = numpy.dstack((fl_y, fl_y, fl_y))

    fl_x_s = numpy.dstack((fl_x_masked, fl_x_masked, fl_x_masked))
    fl_y_s = numpy.dstack((fl_y_masked, fl_y_masked, fl_y_masked))

    mf = numpy.dstack((mask_frame_uint, mask_frame_uint, mask_frame_uint))

    fmasked_crop = Utilities.crop_to_bb_and_resize(fmasked, target_mcbb, fr.shape[:2], fr.shape[:2])
    fl_x_s_crop = Utilities.crop_to_bb_and_resize(fl_x_s, target_mcbb, fr.shape[:2], fr.shape[:2])
    fl_y_s_crop = Utilities.crop_to_bb_and_resize(fl_y_s, target_mcbb, fr.shape[:2], fr.shape[:2])

    fr = cv2.rectangle(fr, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    fl_x = cv2.rectangle(fl_x, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    fl_y = cv2.rectangle(fl_y, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
    mf = cv2.rectangle(mf, (int(target_mcbb[0]), int(target_mcbb[1])), (int(target_mcbb[2]), int(target_mcbb[3])), (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    frshow = numpy.hstack((fr, mf, fmasked, fmasked_crop))
    flxshow = numpy.hstack((fl_x, mf, fl_x_s, fl_x_s_crop))
    flyshow = numpy.hstack((fl_y, mf, fl_y_s, fl_y_s_crop))

    fshow_mock = numpy.vstack(
        (
            # frshow,
            flxshow,
            flyshow,
            # flxyrgb
        )
    )

    cv2.imshow('frame', numpy.vstack((fshow, fshow_mock)))

    frame_data = decoder.read()

    print("MSE -> ", ((fshow - fshow_mock)**2).mean(axis=None), 1/((time.perf_counter()-start_time)), end="\n")  # type: ignore
    print("MSE -> ", ((fshow - fshow_mock)**2).mean(axis=None), 1/((time.perf_counter()-start_time)), end="\r")  # type: ignore

    # # the 'q' button is set as the
    # # quitting button you may use any
    # # desired button of your choice


# After the loop release the cap object
# Destroy all the windows
decoder.stop()
cv2.destroyAllWindows()
mve_load.close()
print(counter, f"{round((101-counter)/101, 1)}% loss")
