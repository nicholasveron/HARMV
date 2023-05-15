import cv2
import numpy
import time
import h5py
from copy import deepcopy
from numpy import ndarray
from importables.utilities import Utilities
from importables.video_decoder import VideoDecoderProcessSpawner
from importables.custom_types import DecodedData, FrameRGB, OpticalFlowFrame
from importables.optical_flow_generator import OpticalFlowGenerator, OpticalFlowFrame, OpticalFlowGeneratorMocker

webcam_ip = "rtsp://NicholasXPS17:8554/cam"
video_path = "/mnt/c/Skripsi/dataset-h264/R002A120/S018C001P008R002A120_rgb.mp4"

# ----------- NORMAL MODE
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
last_frame_data: DecodedData = deepcopy(frame_data)

assert frame_data[0], "DECODER ERROR, ABORTING..."

args_opgenerator = {
    "model_type": "flowformer",
    "model_pretrained": "sintel",
    "bound":  32,
    "raw_optical_flows": False,
    "target_size": 320,
    "overlap_grid_mode": True,
    "overlap_grid_scale": 2
}

op_generator = OpticalFlowGenerator(**args_opgenerator)

while(True):

    start_time = time.perf_counter()
    # Capture the video frame by frame

    if cv2.waitKey(1) & 0xFF == ord('q') or not frame_data[0]:
        decoder.stop()
        break

    last_rgb_frame: FrameRGB = last_frame_data[1]
    rgb_frame: FrameRGB = frame_data[1]

    op_frame: OpticalFlowFrame = op_generator.forward_once_auto(last_rgb_frame, rgb_frame)
    rgb_frame = cv2.resize(rgb_frame, (op_frame.shape[1], op_frame.shape[0]))

    fl_x = op_frame[:, :, 0]
    fl_y = op_frame[:, :, 1]

    # Display the resulting frame

    fl_x_s = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y_s = numpy.dstack((fl_y, fl_y, fl_y))

    stacked = numpy.hstack((rgb_frame,
                            fl_x_s,
                            fl_y_s))

    cv2.imshow('frame', stacked)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    last_frame_data = frame_data
    frame_data = decoder.read()
    print(1/((time.perf_counter()-start_time)), end="\n")
    print(1/((time.perf_counter()-start_time)), end="\r")

# After the loop release the cap object
# Destroy all the windows
decoder.stop()
cv2.destroyAllWindows()

# ----------- SAVING
decoder = VideoDecoderProcessSpawner(**args_decoder).start()
frame_data: DecodedData = decoder.read()
last_frame_data: DecodedData = deepcopy(frame_data)

assert frame_data[0], "DECODER ERROR, ABORTING..."

args_opgenerator = {
    "model_type": "flowformer",
    "model_pretrained": "sintel",
    "bound":  32,
    "raw_optical_flows": True,
    "target_size": 320,
    "overlap_grid_mode": True,
    "overlap_grid_scale": 2
}

op_generator = OpticalFlowGenerator(**args_opgenerator)

op_save = h5py.File("try_mve.h5", mode="w")
mock_op = OpticalFlowGeneratorMocker(op_save, **args_opgenerator)

while(True):

    start_time = time.perf_counter()
    # Capture the video frame by frame

    if cv2.waitKey(1) & 0xFF == ord('q') or not frame_data[0]:
        decoder.stop()
        break

    last_rgb_frame: FrameRGB = last_frame_data[1]
    rgb_frame: FrameRGB = frame_data[1]

    op_frame: OpticalFlowFrame = op_generator.forward_once_auto(last_rgb_frame, rgb_frame)
    rgb_frame = cv2.resize(rgb_frame, (op_frame.shape[1], op_frame.shape[0]))
    mock_op.append(op_frame)
    op_frame = Utilities.bound_motion_frame(op_frame.copy(), 128, 255/(2*args_opgenerator["bound"]))

    fl_x = op_frame[:, :, 0]
    fl_y = op_frame[:, :, 1]

    # Display the resulting frame

    fl_x_s = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y_s = numpy.dstack((fl_y, fl_y, fl_y))

    stacked = numpy.hstack((rgb_frame,
                            fl_x_s,
                            fl_y_s))

    cv2.imshow('frame', stacked)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    last_frame_data = frame_data
    frame_data = decoder.read()
    print(1/((time.perf_counter()-start_time)), end="\n")
    print(1/((time.perf_counter()-start_time)), end="\r")

# After the loop release the cap object
# Destroy all the windows
decoder.stop()
cv2.destroyAllWindows()
mock_op.save()
op_save.close()

# ----------- LOADING
decoder = VideoDecoderProcessSpawner(**args_decoder).start()
frame_data: DecodedData = decoder.read()

assert frame_data[0], "DECODER ERROR, ABORTING..."

args_opgenerator = {
    "model_type": "flowformer",
    "model_pretrained": "sintel",
    "bound":  32,
    "raw_optical_flows": False,
    "target_size": 320,
    "overlap_grid_mode": True,
    "overlap_grid_scale": 2,
}


op_load = h5py.File("try_mve.h5", mode="r")
op_generator = OpticalFlowGenerator(**args_opgenerator)
op_generator_mock = OpticalFlowGeneratorMocker(op_load, **args_opgenerator)
op_generator_mock.load()

while(True):

    start_time = time.perf_counter()
    # Capture the video frame by frame

    if cv2.waitKey(1) & 0xFF == ord('q') or not frame_data[0]:
        decoder.stop()
        break

    last_rgb_frame: FrameRGB = last_frame_data[1]
    rgb_frame: FrameRGB = frame_data[1]

    op_frame: OpticalFlowFrame = op_generator.forward_once_auto(last_rgb_frame, rgb_frame)
    rgb_frame = cv2.resize(rgb_frame, (op_frame.shape[1], op_frame.shape[0]))
    op_frame_mock: OpticalFlowFrame = op_generator_mock.forward_once_auto(last_rgb_frame, rgb_frame)

    fl_x = op_frame[:, :, 0]
    fl_y = op_frame[:, :, 1]

    fl_x_mock = op_frame_mock[:, :, 0]
    fl_y_mock = op_frame_mock[:, :, 1]

    # Display the resulting frame

    fl_x_s = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y_s = numpy.dstack((fl_y, fl_y, fl_y))

    fl_x_s_mock = numpy.dstack((fl_x_mock, fl_x_mock, fl_x_mock))
    fl_y_s_mock = numpy.dstack((fl_y_mock, fl_y_mock, fl_y_mock))

    stacked = numpy.hstack((rgb_frame,
                            fl_x_s,
                            fl_y_s))

    stacked_mock = numpy.hstack((rgb_frame,
                                 fl_x_s_mock,
                                 fl_y_s_mock))

    cv2.imshow('frame', numpy.vstack((stacked, stacked_mock)))

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    last_frame_data = frame_data
    frame_data = decoder.read()
    print("MSE -> ", ((stacked - stacked_mock)**2).mean(axis=None), 1/((time.perf_counter()-start_time)), end="\n")  # type: ignore
    print("MSE -> ", ((stacked - stacked_mock)**2).mean(axis=None), 1/((time.perf_counter()-start_time)), end="\r")  # type: ignore

decoder.stop()
cv2.destroyAllWindows()
op_load.close()
