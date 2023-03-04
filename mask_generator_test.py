
# run rtsp
# https://github.com/aler9/rtsp-simple-server#from-a-webcam
# import the opencv library
import cv2
import numpy
from importables.motion_vector_extractor import MotionVectorExtractorProcessSpawner, MotionVectorExtractor, MotionVectorMocker
from importables.mask_generator import MaskGenerator, MaskMocker
import time
from collections import deque
from ptlflow.utils import flow_utils
import h5py

timer_avg = deque([0.]*101, maxlen=101)

# decoder = MotionVectorExtractor(**args_mvex)
# decoder = MotionVectorExtractorProcessSpawner(**args_mvex, update_rate=200).start()
# decoder = MotionVectorExtractorProcessSpawner("rtsp://192.168.0.101:8554/cam", 15, 30, 30, True, 320, box=True).start()
# decoder = MotionVectorExtractorProcessSpawner("rtsp://192.168.0.183:8554/cam", 15, 30, 30, True, 320, box=True).start()
# decoder = MotionVectorExtractorProcessSpawner("rtsp://192.168.0.185:5540/ch0", 15, 40, 320, True, 320, box=True).start()
# decoder = MotionVectorExtractorProcessSpawner("/mnt/c/Skripsi/dataset-h264/R002A120/S018C001P008R002A120_rgb.mp4", 15, 5, 5, True, 640, box=True).start()

webcam_ip = "rtsp://0.tcp.ap.ngrok.io:14720/ch0"
webcam_ip = "rtsp://192.168.0.101:8554/cam"
video_path = "/mnt/c/Skripsi/dataset-h264/R002A120/S018C001P008R002A120_rgb.mp4"

# ----------- NORMAL RUN
args_mvex = {
    "path":  webcam_ip,
    "bound":  32,
    "raw_motion_vectors": False,
    "camera_realtime": True,
    "camera_update_rate":  60,
    "camera_buffer_size":  0,
    "letterboxed": True,
    "new_shape": 640,
    "box": False,
    "color": (114, 114, 114, 128, 128),
    "stride":  32,
}

yolo_maskgen = MaskGenerator(
    './libs/yolov7-mask/yolov7-mask.pt',
    './libs/yolov7-mask/data/hyp.scratch.mask.yaml',
    0.5,
    0.45)

sample_frame_reader = MotionVectorExtractor(**args_mvex)
first_frame = sample_frame_reader.read()
print(first_frame[1].shape)
sample_frame_reader.stop()
yolo_maskgen.forward_once_maskonly(first_frame[1])

decoder = MotionVectorExtractorProcessSpawner(**args_mvex, update_rate=60).start()

counter = 0

target_mcbb = numpy.array((0, 0, first_frame[1].shape[1], first_frame[1].shape[0])).astype(numpy.float16)

while(True):
    # Capture the video frame by frame
    start_time = time.perf_counter()

    # using constant multiprocessing(faster)
    data = decoder.read()
    available, fr, fl = data

    counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q') or not available:
        decoder.stop()
        break

    # mask_data = yolo_maskgen.forward_once_maskonly(fr)
    mask_data = yolo_maskgen.forward_once_with_mcbb(fr)
    # fl = MotionVectorExtractor.rescale_mv(fl, 128, 1/2*args_mvex["bound"])

    fl_x = fl[..., 0]
    fl_y = fl[..., 1]

    # letterbox no need, decoder with letterbox enabled
    # fr = rev_letterbox(fr, 320)
    # fl_x = rev_letterbox(fl_x, 320)
    # fl_y = rev_letterbox(fl_y, 320)

    # using threading, (slower, avg 48fps, 1% 30fps )
    # data = decoder.read_while_process(yolo_maskgen.generate_once)
    # available, fr, fl = data[0]

    # if cv2.waitKey(1) & 0xFF == ord('q') or not available:
    #     decoder.stop()
    #     break

    # fl_x = fl[..., 0]
    # fl_y = fl[..., 1]
    # mask_data = data[1]

    mcbb = mask_data[2]
    target_mcbb = (target_mcbb + (mcbb.astype(numpy.float16) - target_mcbb) * 0.1)
    mask_data = mask_data[1]

    # fr = cv2.rectangle(fr, (int(mcbb[0]), int(mcbb[1])), (int(mcbb[2]), int(mcbb[3])), (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)

    mask_data_uint = (mask_data.astype(numpy.uint8)*255)

    combine_to_mask = numpy.dstack(
        (
            numpy.copy(fr),
            numpy.copy(fl_x),
            numpy.copy(fl_y)
        )
    )

    combine_to_mask[~mask_data] = (255, 255, 255, 128, 128)
    fr = MaskGenerator.crop_to_bb_and_rescale(fr, target_mcbb)

    timer_avg.append(1/((time.perf_counter()-start_time)))
    tl_avg = list(timer_avg)
    tl_avg.sort()
    tl_avg_10 = tl_avg[:len(timer_avg)//10]
    tl_avg_1 = tl_avg[:len(timer_avg)//100]
    print(sum(timer_avg)//len(timer_avg), sum(tl_avg_10)//len(tl_avg_10), sum(tl_avg_1)//len(tl_avg_1), end="\r")

    fmasked = combine_to_mask[:, :, 0:3]
    fl_x_masked = combine_to_mask[:, :, 3]
    fl_y_masked = combine_to_mask[:, :, 4]

    fl_x = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y = numpy.dstack((fl_y, fl_y, fl_y))

    fl_x_s = numpy.dstack((fl_x_masked, fl_x_masked, fl_x_masked))
    fl_y_s = numpy.dstack((fl_y_masked, fl_y_masked, fl_y_masked))

    mf = numpy.dstack((mask_data_uint, mask_data_uint, mask_data_uint))

    frshow = numpy.hstack((fr, mf, fmasked))
    flxshow = numpy.hstack((fl_x, mf, fl_x_s))
    flyshow = numpy.hstack((fl_y, mf, fl_y_s))

    flxy_masked = flow_utils.flow_to_rgb(numpy.dstack((fl_x_masked, fl_y_masked)))
    flxy_masked = cv2.cvtColor(flxy_masked, cv2.COLOR_RGB2BGR)  # type: ignore

    flxyrgb = numpy.hstack((fl_x_s, fl_y_s, flxy_masked))

    fshow = numpy.vstack(
        (
            frshow,
            flxshow,
            flyshow,
            flxyrgb
        )
    )

    cv2.imshow('frame', fshow)

    # # the 'q' button is set as the
    # # quitting button you may use any
    # # desired button of your choice


# After the loop release the cap object
# Destroy all the windows
decoder.stop()
cv2.destroyAllWindows()

exit()

webcam_ip = "rtsp://192.168.0.101:8554/cam"
video_path = "/mnt/c/Skripsi/dataset-h264/R002A120/S018C001P008R002A120_rgb.mp4"

# ----------- SAVING
args_mvex = {
    "path":  video_path,
    "bound":  32,
    "raw_motion_vectors": True,
    "camera_realtime": False,
    "camera_update_rate":  300,
    "camera_buffer_size":  0,
    "letterboxed": True,
    "new_shape": 640,
    "box": False,
    "color": (114, 114, 114, 128, 128),
    "stride":  32,
}

yolo_maskgen = MaskGenerator(
    './libs/yolov7-mask/yolov7-mask.pt',
    './libs/yolov7-mask/data/hyp.scratch.mask.yaml',
    0.5,
    0.45)

sample_frame_reader = MotionVectorExtractor(**args_mvex)
first_frame = sample_frame_reader.read()
print(first_frame[1].shape)
sample_frame_reader.stop()
yolo_maskgen.forward_once_maskonly(first_frame[1])

decoder = MotionVectorExtractorProcessSpawner(**args_mvex, update_rate=300).start()
mve_save = h5py.File("try_mve.h5", mode="w")
mock_mve = MotionVectorMocker(mve_save, **args_mvex)
mock_maskgen = MaskMocker(mve_save, **args_mvex)

counter = 0

while(True):
    # Capture the video frame by frame
    start_time = time.perf_counter()

    # using constant multiprocessing(faster)
    data = decoder.read()
    mock_mve.append(data)
    available, fr, fl = data

    counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q') or not available:
        decoder.stop()
        break

    mask_data = yolo_maskgen.forward_once_with_mcbb(fr)
    mock_maskgen.append(mask_data)

    fl = MotionVectorExtractor.rescale_mv(fl, 128, 1/2*args_mvex["bound"])

    fl_x = fl[..., 0]
    fl_y = fl[..., 1]

    # letterbox no need, decoder with letterbox enabled
    # fr = rev_letterbox(fr, 320)
    # fl_x = rev_letterbox(fl_x, 320)
    # fl_y = rev_letterbox(fl_y, 320)

    # using threading, (slower, avg 48fps, 1% 30fps )
    # data = decoder.read_while_process(yolo_maskgen.generate_once)
    # available, fr, fl = data[0]

    # if cv2.waitKey(1) & 0xFF == ord('q') or not available:
    #     decoder.stop()
    #     break

    # fl_x = fl[..., 0]
    # fl_y = fl[..., 1]
    # mask_data = data[1]

    mcbb = mask_data[2]
    mask_data = mask_data[1]

    mask_data_uint = (mask_data.astype(numpy.uint8)*255)

    combine_to_mask = numpy.dstack(
        (
            numpy.copy(fr),
            numpy.copy(fl_x),
            numpy.copy(fl_y)
        )
    )

    combine_to_mask[~mask_data] = (255, 255, 255, 128, 128)

    timer_avg.append(1/((time.perf_counter()-start_time)))
    tl_avg = list(timer_avg)
    tl_avg.sort()
    tl_avg_10 = tl_avg[:len(timer_avg)//10]
    tl_avg_1 = tl_avg[:len(timer_avg)//100]
    print(sum(timer_avg)//len(timer_avg), sum(tl_avg_10)//len(tl_avg_10), sum(tl_avg_1)//len(tl_avg_1), end="\r")

    fmasked = combine_to_mask[:, :, 0:3]
    fl_x_masked = combine_to_mask[:, :, 3]
    fl_y_masked = combine_to_mask[:, :, 4]

    fl_x = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y = numpy.dstack((fl_y, fl_y, fl_y))

    fl_x_s = numpy.dstack((fl_x_masked, fl_x_masked, fl_x_masked))
    fl_y_s = numpy.dstack((fl_y_masked, fl_y_masked, fl_y_masked))

    mf = numpy.dstack((mask_data_uint, mask_data_uint, mask_data_uint))

    fr = MaskGenerator.crop_to_bb_and_rescale(fr, mcbb)

    frshow = numpy.hstack((fr, mf, fmasked))
    flxshow = numpy.hstack((fl_x, mf, fl_x_s))
    flyshow = numpy.hstack((fl_y, mf, fl_y_s))

    flxy_masked = flow_utils.flow_to_rgb(numpy.dstack((fl_x_masked, fl_y_masked)))
    flxy_masked = cv2.cvtColor(flxy_masked, cv2.COLOR_RGB2BGR)  # type: ignore

    flxyrgb = numpy.hstack((fl_x_s, fl_y_s, flxy_masked))

    fshow = numpy.vstack(
        (
            frshow,
            flxshow,
            flyshow,
            flxyrgb
        )
    )

    cv2.imshow('frame', fshow)

    # # the 'q' button is set as the
    # # quitting button you may use any
    # # desired button of your choice


# After the loop release the cap object
# Destroy all the windows
decoder.stop()
cv2.destroyAllWindows()
mock_mve.save()
mock_maskgen.save()
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
args_mvex = {
    "path":  video_path,
    "bound":  32,
    "raw_motion_vectors": True,
    "camera_realtime": False,
    "camera_update_rate":  300,
    "camera_buffer_size":  0,
    "letterboxed": True,
    "new_shape": 640,
    "box": False,
    "color": (114, 114, 114, 128, 128),
    "stride":  32,
}

mve_load = h5py.File("try_mve.h5", mode="r")

yolo_maskgen = MaskMocker(
    mve_load,
    './libs/yolov7-mask/yolov7-mask.pt',
    './libs/yolov7-mask/data/hyp.scratch.mask.yaml',
    0.5,
    0.45)

decoder = MotionVectorMocker(mve_load, **args_mvex, update_rate=300).start()
yolo_maskgen.load()
decoder.load()

counter = 0

while(True):
    # Capture the video frame by frame
    start_time = time.perf_counter()

    # using constant multiprocessing(faster)
    data = decoder.read()
    available, fr, fl = data

    counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q') or not available:
        decoder.stop()
        break

    mask_data = yolo_maskgen.forward_once_with_mcbb(fr)

    fl = MotionVectorExtractor.rescale_mv(fl, 128, 1/2*args_mvex["bound"])

    fl_x = fl[..., 0]
    fl_y = fl[..., 1]

    # letterbox no need, decoder with letterbox enabled
    # fr = rev_letterbox(fr, 320)
    # fl_x = rev_letterbox(fl_x, 320)
    # fl_y = rev_letterbox(fl_y, 320)

    # using threading, (slower, avg 48fps, 1% 30fps )
    # data = decoder.read_while_process(yolo_maskgen.generate_once)
    # available, fr, fl = data[0]

    # if cv2.waitKey(1) & 0xFF == ord('q') or not available:
    #     decoder.stop()
    #     break

    # fl_x = fl[..., 0]
    # fl_y = fl[..., 1]
    # mask_data = data[1]

    mcbb = mask_data[2]
    mask_data = mask_data[1]

    mask_data_uint = (mask_data.astype(numpy.uint8)*255)

    combine_to_mask = numpy.dstack(
        (
            numpy.copy(fr),
            numpy.copy(fl_x),
            numpy.copy(fl_y)
        )
    )

    combine_to_mask[~mask_data] = (255, 255, 255, 128, 128)

    timer_avg.append(1/((time.perf_counter()-start_time)))
    tl_avg = list(timer_avg)
    tl_avg.sort()
    tl_avg_10 = tl_avg[:len(timer_avg)//10]
    tl_avg_1 = tl_avg[:len(timer_avg)//100]
    print(sum(timer_avg)//len(timer_avg), sum(tl_avg_10)//len(tl_avg_10), sum(tl_avg_1)//len(tl_avg_1), end="\r")

    fmasked = combine_to_mask[:, :, 0:3]
    fl_x_masked = combine_to_mask[:, :, 3]
    fl_y_masked = combine_to_mask[:, :, 4]

    fl_x = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y = numpy.dstack((fl_y, fl_y, fl_y))

    fl_x_s = numpy.dstack((fl_x_masked, fl_x_masked, fl_x_masked))
    fl_y_s = numpy.dstack((fl_y_masked, fl_y_masked, fl_y_masked))

    mf = numpy.dstack((mask_data_uint, mask_data_uint, mask_data_uint))

    fr = MaskGenerator.crop_to_bb_and_rescale(fr, mcbb)

    frshow = numpy.hstack((fr, mf, fmasked))
    flxshow = numpy.hstack((fl_x, mf, fl_x_s))
    flyshow = numpy.hstack((fl_y, mf, fl_y_s))

    flxy_masked = flow_utils.flow_to_rgb(numpy.dstack((fl_x_masked, fl_y_masked)))
    flxy_masked = cv2.cvtColor(flxy_masked, cv2.COLOR_RGB2BGR)  # type:ignore

    flxyrgb = numpy.hstack((fl_x_s, fl_y_s, flxy_masked))

    fshow = numpy.vstack(
        (
            frshow,
            flxshow,
            flyshow,
            flxyrgb
        )
    )

    cv2.imshow('frame', fshow)

    # # the 'q' button is set as the
    # # quitting button you may use any
    # # desired button of your choice


# After the loop release the cap object
# Destroy all the windows
decoder.stop()
cv2.destroyAllWindows()
mve_load.close()
print(counter, f"{round((101-counter)/101, 1)}% loss")
