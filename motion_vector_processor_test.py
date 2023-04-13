
# run rtsp
# https://github.com/aler9/rtsp-simple-server#from-a-webcam
# import the opencv library
import cv2
import numpy
import time
import h5py
from numpy import ndarray
from importables.utilities import Utilities
from importables.video_decoder import VideoDecoderProcessSpawner
from importables.custom_types import DecodedData, FrameRGB, RawMotionVectors, MotionVectorFrame
from importables.motion_vector_processor import MotionVectorProcessor, MotionVectorFrame, MotionVectorProcessorMocker

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

assert frame_data[0], "DECODER ERROR, ABORTING..."

args_mvprocessor = {
    "input_size": frame_data[1].shape[:2],
    "bound":  32,
    "raw_motion_vectors": False,
    "target_size": 320,
}

mv_processor = MotionVectorProcessor(**args_mvprocessor)

while(True):

    start_time = time.perf_counter()
    # Capture the video frame by frame

    if cv2.waitKey(1) & 0xFF == ord('q') or not frame_data[0]:
        decoder.stop()
        break

    rgb_frame: FrameRGB = frame_data[1]
    raw_mv: RawMotionVectors = frame_data[2]

    mv_frame: MotionVectorFrame = mv_processor.process(raw_mv)
    rgb_frame = cv2.resize(rgb_frame, (mv_frame.shape[1], mv_frame.shape[0]))

    fl_x = mv_frame[:, :, 0]
    fl_y = mv_frame[:, :, 1]

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

assert frame_data[0], "DECODER ERROR, ABORTING..."

args_mvprocessor = {
    "input_size": frame_data[1].shape[:2],
    "bound":  32,
    "raw_motion_vectors": True,
}

mv_processor = MotionVectorProcessor(**args_mvprocessor)

mve_save = h5py.File("try_mve.h5", mode="w")
mock_mve = MotionVectorProcessorMocker(mve_save, **args_mvprocessor)

while(True):

    start_time = time.perf_counter()
    # Capture the video frame by frame

    if cv2.waitKey(1) & 0xFF == ord('q') or not frame_data[0]:
        decoder.stop()
        break

    rgb_frame: FrameRGB = frame_data[1]
    raw_mv: RawMotionVectors = frame_data[2]

    mv_frame: MotionVectorFrame = mv_processor.process(raw_mv)
    rgb_frame = cv2.resize(rgb_frame, (mv_frame.shape[1], mv_frame.shape[0]))
    mock_mve.append(mv_frame)
    mv_frame = Utilities.bound_motion_frame(mv_frame.copy(), 128, 255/(2*args_mvprocessor["bound"]))

    fl_x = mv_frame[:, :, 0]
    fl_y = mv_frame[:, :, 1]

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

    frame_data = decoder.read()
    print(1/((time.perf_counter()-start_time)), end="\n")
    print(1/((time.perf_counter()-start_time)), end="\r")

# After the loop release the cap object
# Destroy all the windows
decoder.stop()
cv2.destroyAllWindows()
mock_mve.save()
mve_save.close()

# ----------- LOADING
decoder = VideoDecoderProcessSpawner(**args_decoder).start()
frame_data: DecodedData = decoder.read()

assert frame_data[0], "DECODER ERROR, ABORTING..."

args_mvprocessor = {
    "input_size": frame_data[1].shape[:2],
    "bound":  32,
    "raw_motion_vectors": False,
}


mve_load = h5py.File("try_mve.h5", mode="r")
mv_processor = MotionVectorProcessor(**args_mvprocessor)
mv_processor_mock = MotionVectorProcessorMocker(mve_load, **args_mvprocessor)
mv_processor_mock.load()

while(True):

    start_time = time.perf_counter()
    # Capture the video frame by frame

    if cv2.waitKey(1) & 0xFF == ord('q') or not frame_data[0]:
        decoder.stop()
        break

    rgb_frame: FrameRGB = frame_data[1]
    raw_mv: RawMotionVectors = frame_data[2]

    mv_frame: MotionVectorFrame = mv_processor.process(raw_mv)
    rgb_frame = cv2.resize(rgb_frame, (mv_frame.shape[1], mv_frame.shape[0]))
    mv_frame_mock: MotionVectorFrame = mv_processor_mock.process(raw_mv)

    fl_x = mv_frame[:, :, 0]
    fl_y = mv_frame[:, :, 1]

    fl_x_mock = mv_frame_mock[:, :, 0]
    fl_y_mock = mv_frame_mock[:, :, 1]

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

    frame_data = decoder.read()
    print("MSE -> ", ((stacked - stacked_mock)**2).mean(axis=None), 1/((time.perf_counter()-start_time)), end="\n")  # type: ignore
    print("MSE -> ", ((stacked - stacked_mock)**2).mean(axis=None), 1/((time.perf_counter()-start_time)), end="\r")  # type: ignore

decoder.stop()
cv2.destroyAllWindows()
mve_load.close()
