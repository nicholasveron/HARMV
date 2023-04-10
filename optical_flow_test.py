
# run rtsp
# https://github.com/aler9/rtsp-simple-server#from-a-webcam
# import the opencv library
import cv2
import numpy
from importables.mask_generator import MaskGenerator, MaskMocker
from importables.optical_flow_generator import OpticalFlowGenerator, OpticalFlowMocker
from importables.motion_vector_extractor import MotionVectorExtractor, MotionVectorMocker
import time
import h5py
import torch


webcam_ip = "rtsp://NicholasXPS17:8554/cam"
# webcam_ip = "rtsp://192.168.0.122:8554/cam"
video_path = "/mnt/c/Skripsi/dataset-h264/R002A120/S018C001P008R002A120_rgb.mp4"

# ----------- NORMAL MODE
args_mvex = {
    "path":  video_path,
    "bound":  32,
    "raw_motion_vectors": False,
    "camera_realtime": False,
    "camera_update_rate":  120,
    "camera_buffer_size":  0,
}

args_mvex = {
    "path":  webcam_ip,
    "bound":  32,
    "raw_motion_vectors": False,
    "camera_realtime": True,
    "camera_update_rate":  120,
    "camera_buffer_size":  0,
}

args_yolo = {
    "weight_path": './libs/yolov7-mask/yolov7-mask.pt',
    "hyperparameter_path": './libs/yolov7-mask/data/hyp.scratch.mask.yaml',
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "target_input_size": 320,
    "optimize_model": True,
}


yolo_maskgen = MaskGenerator(**args_yolo)

args_opgen = {
    "model_type": "raft_small",
    "model_pretrained": "things",
    "bound": 32,
    "raw_optical_flows": False,
}

# opgen = OpticalFlowGenerator("flowformer", "sintel", 64)
opgen = OpticalFlowGenerator(**args_opgen)

sample_frame_reader = MotionVectorExtractor(**args_mvex)
first_frame = sample_frame_reader.read()
second_frame = sample_frame_reader.read()
print(first_frame[1].shape)
sample_frame_reader.stop()

# opgen.forward_once(first_frame[1], second_frame[1])
yolo_maskgen.forward_once_maskonly(first_frame[1])

decoder = MotionVectorExtractor(**args_mvex)
last_fr = numpy.empty((1))
while(True):

    start_time = time.perf_counter()
    # Capture the video frame by frame
    data = decoder.read()
    available, fr, fl = data

    fr = cv2.resize(fr, (320, 180))
    fl = cv2.resize(fl, (320, 180))

    if cv2.waitKey(1) & 0xFF == ord('q') or not data[0]:
        decoder.stop()
        break

    mask_data = yolo_maskgen.forward_once_with_mcbb(fr)

    if len(last_fr.shape) == 1:
        last_fr = fr.copy()

    op_fl = opgen.forward_once_with_overlap_grid(last_fr, fr, 2)
    op_fl_no_grid = opgen.forward_once(last_fr, fr)
    last_fr = fr.copy()

    op_fl_no_grid_x = op_fl_no_grid[..., 0]
    op_fl_no_grid_y = op_fl_no_grid[..., 1]

    op_fl_x = op_fl[..., 0]
    op_fl_y = op_fl[..., 1]

    fl_x = fl[..., 0]
    fl_y = fl[..., 1]

    mcbb = mask_data[2]
    mask_data = mask_data[1]
    mask_data_uint = (mask_data.astype(numpy.uint8)*255)

    mf = numpy.dstack((mask_data_uint, mask_data_uint, mask_data_uint))

    fr = MaskGenerator.crop_to_bb_and_resize(fr, mcbb, fr.shape[:2], fr.shape[:2])

    # Display the resulting frame

    op_fl_x_s = numpy.dstack((op_fl_x, op_fl_x, op_fl_x))
    op_fl_y_s = numpy.dstack((op_fl_y, op_fl_y, op_fl_y))

    stacked_op = numpy.hstack((fr,
                               op_fl_x_s,
                               op_fl_y_s))

    op_fl_no_grid_x_s = numpy.dstack((op_fl_no_grid_x, op_fl_no_grid_x, op_fl_no_grid_x))
    op_fl_no_grid_y_s = numpy.dstack((op_fl_no_grid_y, op_fl_no_grid_y, op_fl_no_grid_y))

    stacked_no_grid_op = numpy.hstack((mf,
                                       op_fl_no_grid_x_s,
                                       op_fl_no_grid_y_s))

    fl_x_s = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y_s = numpy.dstack((fl_y, fl_y, fl_y))

    stacked = numpy.hstack((fr,
                            fl_x_s,
                            fl_y_s))

    cv2.imshow('frame', numpy.vstack((stacked, stacked_op, stacked_no_grid_op)))

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    print(1/((time.perf_counter()-start_time)), end="\r")


# After the loop release the cap object
# decoder.stop()
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
    "camera_update_rate":  120,
    "camera_buffer_size":  0,
}

args_mvex = {
    "path":  webcam_ip,
    "bound":  32,
    "raw_motion_vectors": True,
    "camera_realtime": True,
    "camera_update_rate":  120,
    "camera_buffer_size":  0,
}

args_yolo = {
    "weight_path": './libs/yolov7-mask/yolov7-mask.pt',
    "hyperparameter_path": './libs/yolov7-mask/data/hyp.scratch.mask.yaml',
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "target_input_size": 320,
    "optimize_model": True,
}


yolo_maskgen = MaskGenerator(**args_yolo)

args_opgen = {
    "model_type": "raft_small",
    "model_pretrained": "things",
    "bound": 32,
    "raw_optical_flows": True,
}

# opgen = OpticalFlowGenerator("flowformer", "sintel", 64)
opgen = OpticalFlowGenerator(**args_opgen)

sample_frame_reader = MotionVectorExtractor(**args_mvex)
first_frame = sample_frame_reader.read()
second_frame = sample_frame_reader.read()
print(first_frame[1].shape)
sample_frame_reader.stop()

opgen.forward_once(first_frame[1], second_frame[1])
yolo_maskgen.forward_once_maskonly(first_frame[1])

mve_save = h5py.File("try_mve.h5", mode="w")
mock_mve = MotionVectorMocker(mve_save, **args_mvex)
mock_of = OpticalFlowMocker(mve_save, **args_mvex)
mock_maskgen = MaskMocker(mve_save, **args_mvex)

decoder = MotionVectorExtractor(**args_mvex)
last_fr = numpy.empty((1))
while(True):

    start_time = time.perf_counter()
    # Capture the video frame by frame
    data = decoder.read()
    mock_mve.append(data)
    available, fr, fl = data

    if cv2.waitKey(1) & 0xFF == ord('q') or not data[0]:
        decoder.stop()
        break

    mask_data = yolo_maskgen.forward_once_with_mcbb(fr)
    mock_maskgen.append(mask_data)

    if len(last_fr.shape) == 1:
        last_fr = fr.copy()

    op_fl = opgen.forward_once_with_overlap_grid(last_fr, fr)
    op_fl_no_grid = opgen.forward_once(last_fr, fr)
    last_fr = fr.copy()
    mock_of.append(op_fl)

    op_fl_no_grid = MotionVectorExtractor.rescale_mv(op_fl_no_grid, 128, 1/2*args_mvex["bound"])
    op_fl = MotionVectorExtractor.rescale_mv(op_fl, 128, 1/2*args_mvex["bound"])

    op_fl_no_grid_x = op_fl_no_grid[..., 0]
    op_fl_no_grid_y = op_fl_no_grid[..., 1]

    op_fl_x = op_fl[..., 0]
    op_fl_y = op_fl[..., 1]

    fl = MotionVectorExtractor.rescale_mv(fl, 128, 1/2*args_mvex["bound"])

    fl_x = fl[..., 0]
    fl_y = fl[..., 1]

    mcbb = mask_data[2]
    mask_data = mask_data[1]
    mask_data_uint = (mask_data.astype(numpy.uint8)*255)

    mf = numpy.dstack((mask_data_uint, mask_data_uint, mask_data_uint))

    fr = MaskGenerator.crop_to_bb_and_resize(fr, mcbb)

    # Display the resulting frame

    op_fl_x_s = numpy.dstack((op_fl_x, op_fl_x, op_fl_x))
    op_fl_y_s = numpy.dstack((op_fl_y, op_fl_y, op_fl_y))

    stacked_op = numpy.hstack((fr,
                               op_fl_x_s,
                               op_fl_y_s))

    op_fl_no_grid_x_s = numpy.dstack((op_fl_no_grid_x, op_fl_no_grid_x, op_fl_no_grid_x))
    op_fl_no_grid_y_s = numpy.dstack((op_fl_no_grid_y, op_fl_no_grid_y, op_fl_no_grid_y))

    stacked_no_grid_op = numpy.hstack((mf,
                                       op_fl_no_grid_x_s,
                                       op_fl_no_grid_y_s))

    fl_x_s = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y_s = numpy.dstack((fl_y, fl_y, fl_y))

    stacked = numpy.hstack((fr,
                            fl_x_s,
                            fl_y_s))

    cv2.imshow('frame', numpy.vstack((stacked, stacked_op, stacked_no_grid_op)))

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    print(1/((time.perf_counter()-start_time)), end="\r")


# After the loop release the cap object
# decoder.stop()
# Destroy all the windows
decoder.stop()
cv2.destroyAllWindows()
mock_maskgen.save()
mock_mve.save()
mock_of.save()
mve_save.close()

# ----------- LOADING
args_mvex = {
    "path":  video_path,
    "bound":  32,
    "raw_motion_vectors": False,
    "camera_realtime": False,
    "camera_update_rate":  60,
    "camera_buffer_size":  0,
    "letterboxed": True,
    "new_shape": 320,
    "box": False,
    "color": (114, 114, 114, 128, 128),
    "stride":  32,
}

args_opgen = {
    "model_type": "raft_small",
    "model_pretrained": "things",
    "bound": 32,
    "raw_optical_flows": False,
}

mve_load = h5py.File("try_mve.h5", mode="r")

# opgen = OpticalFlowGenerator("flowformer", "sintel", 64)
opgen = OpticalFlowMocker(mve_load, **args_opgen)
yolo_maskgen = MaskMocker(
    mve_load,
    './libs/yolov7-mask/yolov7-mask.pt',
    './libs/yolov7-mask/data/hyp.scratch.mask.yaml',
    0.5,
    0.45)

decoder = MotionVectorMocker(mve_load, **args_mvex, update_rate=300)
opgen.load()
yolo_maskgen.load()
decoder.load()
last_fr = numpy.empty((1))

counter = 0

while(True):

    start_time = time.perf_counter()
    # Capture the video frame by frame
    data = decoder.read()
    available, fr, fl = data

    counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q') or not data[0]:
        decoder.stop()
        break

    mask_data = yolo_maskgen.forward_once_with_mcbb(fr)

    if len(last_fr.shape) == 1:
        last_fr = fr.copy()

    op_fl = opgen.forward_once_with_overlap_grid(last_fr, fr)
    last_fr = fr.copy()

    op_fl_x = op_fl[..., 0]
    op_fl_y = op_fl[..., 1]

    fl_x = fl[..., 0]
    fl_y = fl[..., 1]

    mcbb = mask_data[2]
    mask_data = mask_data[1]
    mask_data_uint = (mask_data.astype(numpy.uint8)*255)

    mf = numpy.dstack((mask_data_uint, mask_data_uint, mask_data_uint))

    fr = MaskGenerator.crop_to_bb_and_resize(fr, mcbb)

    # Display the resulting frame

    op_fl_x_s = numpy.dstack((op_fl_x, op_fl_x, op_fl_x))
    op_fl_y_s = numpy.dstack((op_fl_y, op_fl_y, op_fl_y))

    stacked_op = numpy.hstack((fr,
                               op_fl_x_s,
                               op_fl_y_s))

    fl_x_s = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y_s = numpy.dstack((fl_y, fl_y, fl_y))

    stacked = numpy.hstack((mf,
                            fl_x_s,
                            fl_y_s))

    cv2.imshow('frame', numpy.vstack((stacked, stacked_op)))

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    time.sleep(1/5)

    print(1/((time.perf_counter()-start_time)), end="\r")


# After the loop release the cap object
# decoder.stop()
# Destroy all the windows
decoder.stop()
cv2.destroyAllWindows()
mve_load.close()
print(counter, f"{round((101-counter)/101, 1)}% loss")
