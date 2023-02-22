
# run rtsp
# https://github.com/aler9/rtsp-simple-server#from-a-webcam
# import the opencv library
import cv2
import numpy
from importables.motion_vector_extractor import MotionVectorExtractorProcessSpawner, MotionVectorExtractor
from importables.mask_generator import MaskGenerator
import time
from collections import deque

timer_avg = deque([0]*101, maxlen=101)

yolo_maskgen = MaskGenerator(
    './libs/yolov7-mask/yolov7-mask.pt',
    './libs/yolov7-mask/data/hyp.scratch.mask.yaml',
    0.5,
    0.45)

mv_args = {
    "path": "rtsp://0.tcp.ap.ngrok.io:17426/cam",
    "bound": 15,
    "camera_sampling_rate": 30,
    "letterboxed": True,
    "new_shape": 320,
    "box": False
}

sample_frame_reader = MotionVectorExtractor(**mv_args)
first_frame = sample_frame_reader.read()
print(first_frame[1].shape)
sample_frame_reader.stop()
yolo_maskgen.generate_once(first_frame[1])
decoder = MotionVectorExtractorProcessSpawner(**mv_args, sampling_rate=30).start()
# decoder = MotionVectorExtractorProcessSpawner("rtsp://192.168.0.101:8554/cam", 15, 30, 30, True, 320, box=True).start()
# decoder = MotionVectorExtractorProcessSpawner("rtsp://192.168.0.183:8554/cam", 15, 30, 30, True, 320, box=True).start()
# decoder = MotionVectorExtractorProcessSpawner("rtsp://192.168.0.185:5540/ch0", 15, 40, 320, True, 320, box=True).start()
# decoder = MotionVectorExtractorProcessSpawner("/mnt/c/Skripsi/dataset-h264/R002A120/S018C001P008R002A120_rgb.mp4", 15, 5, 5, True, 640, box=True).start()


while(True):
    # Capture the video frame by frame
    start_time = time.perf_counter()

    data = decoder.read()

    if cv2.waitKey(1) & 0xFF == ord('q') or not data[0]:
        decoder.stop()
        break

    fr = data[1]
    fl_x = data[2]
    fl_y = data[3]

    # letterbox no need, decoder with letterbox enabled
    # fr = rev_letterbox(fr, 320)
    # fl_x = rev_letterbox(fl_x, 320)
    # fl_y = rev_letterbox(fl_y, 320)
    mask_data = yolo_maskgen.generate_once(fr)

    mask_data = mask_data[1]

    mask_data_uint = (mask_data.astype(numpy.uint8)*255)

    combine_to_mask = numpy.dstack(
        (
            numpy.copy(fr),
            numpy.copy(fl_x),
            numpy.copy(fl_y)
        )
    )

    combine_to_mask[~mask_data] = (255, 255, 255, 127, 127)

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

    fshow = numpy.vstack(
        (frshow, flxshow, flyshow)
    )

    cv2.imshow('frame', fshow)

    # # the 'q' button is set as the
    # # quitting button you may use any
    # # desired button of your choice


# After the loop release the cap object
# Destroy all the windows
decoder.stop()
cv2.destroyAllWindows()

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
