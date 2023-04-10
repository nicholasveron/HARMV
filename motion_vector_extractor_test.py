
# run rtsp
# https://github.com/aler9/rtsp-simple-server#from-a-webcam
# import the opencv library
import cv2
import numpy
from importables.motion_vector_extractor import MotionVectorExtractor, MotionVectorMocker
import time
import h5py

# decoder = MotionVectorExtractorProcessSpawner("rtsp://192.168.0.186:8554/cam", 15, 60, 120, True, 320, box=True).start()
# decoder = MotionVectorExtractorProcessSpawner("rtsp://0.tcp.ap.ngrok.io:15225/cam", 15, 30, 30, True, 320).start()
# decoder = MotionVectorExtractorProcessSpawner("rtsp://192.168.0.101:8554/cam", 15, 30, 30, True, 320).start()
# decoder = MotionVectorExtractorProcessSpawner("/mnt/c/Skripsi/dataset-h264/R001A001/S001C001P001R001A001_rgb.mp4", 15, 5, 5, True, 320).start()
# decoder = MotionVectorExtractorProcessSpawner("/mnt/c/Skripsi/dataset-h264-libx/R001A001/S001C001P001R001A001_rgb.mp4", 15, 10, 10, True, 320).start()


webcam_ip = "rtsp://NicholasXPS17:8554/cam"
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

decoder = MotionVectorExtractor(**args_mvex)

while(True):

    start_time = time.perf_counter()
    # Capture the video frame by frame
    data = decoder.read()

    if cv2.waitKey(1) & 0xFF == ord('q') or not data[0]:
        decoder.stop()
        break

    fl = data[2]

    fr = data[1]
    fl_x = fl[:, :, 0]
    fl_y = fl[:, :, 1]

    # Display the resulting frame

    fl_x_s = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y_s = numpy.dstack((fl_y, fl_y, fl_y))

    stacked = numpy.hstack((fr,
                            fl_x_s,
                            fl_y_s))

    cv2.imshow('frame', stacked)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    print(1/((time.perf_counter()-start_time)), end="\r")

# After the loop release the cap object
# decoder.stop()
# Destroy all the windows
cv2.destroyAllWindows()


exit()
webcam_ip = "rtsp://NicholasXPS17:8554/cam"
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

decoder = MotionVectorExtractor(**args_mvex)
mve_save = h5py.File("try_mve.h5", mode="w")
mock_mve = MotionVectorMocker(mve_save, **args_mvex)

while(True):

    start_time = time.perf_counter()
    # Capture the video frame by frame
    data = decoder.read()
    mock_mve.append(data)

    if cv2.waitKey(1) & 0xFF == ord('q') or not data[0]:
        decoder.stop()
        break

    fl = data[2]
    fl = MotionVectorExtractor.rescale_mv(fl.copy(), 128, 1/2*args_mvex["bound"])

    fr = data[1]
    fl_x = fl[:, :, 0]
    fl_y = fl[:, :, 1]

    # Display the resulting frame

    fl_x_s = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y_s = numpy.dstack((fl_y, fl_y, fl_y))

    stacked = numpy.hstack((fr,
                            fl_x_s,
                            fl_y_s))

    cv2.imshow('frame', stacked)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    print(1/((time.perf_counter()-start_time)), end="\r")

# After the loop release the cap object
# decoder.stop()
# Destroy all the windows
cv2.destroyAllWindows()

mock_mve.save()
mve_save.close()

# ----------- LOADING
args_mvex = {
    "path":  video_path,
    "bound":  64,
    "raw_motion_vectors": False,
    "camera_realtime": False,
    "camera_update_rate":  30,
    "camera_buffer_size":  0,
    "letterboxed": True,
    "new_shape": 640,
    "box": False,
    "color": (114, 114, 114, 128, 128),
    "stride":  32,
}  # most of the var will be ignored

mve_load = h5py.File("try_mve.h5", mode="r")
decoder_mock = MotionVectorMocker(mve_load, **args_mvex)
decoder_mock.load()
decoder = MotionVectorExtractor(**args_mvex)

while(True):

    start_time = time.perf_counter()
    # Capture the video frame by frame
    data = decoder.read()
    data_mock = decoder_mock.read()

    if cv2.waitKey(1) & 0xFF == ord('q') or not (data_mock[0] and data[0]):
        decoder.stop()
        decoder_mock.stop()
        break

    fl_mock = data_mock[2]

    fr_mock = data_mock[1]
    fl_x_mock = fl_mock[:, :, 0]
    fl_y_mock = fl_mock[:, :, 1]

    # Display the resulting frame

    fl_x_s_mock = numpy.dstack((fl_x_mock, fl_x_mock, fl_x_mock))
    fl_y_s_mock = numpy.dstack((fl_y_mock, fl_y_mock, fl_y_mock))

    stacked_mock = numpy.hstack((fr_mock,
                                 fl_x_s_mock,
                                 fl_y_s_mock))

    fl = data[2]

    fr = data[1]
    fl_x = fl[:, :, 0]
    fl_y = fl[:, :, 1]

    # Display the resulting frame

    fl_x_s = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y_s = numpy.dstack((fl_y, fl_y, fl_y))

    stacked = numpy.hstack((fr,
                            fl_x_s,
                            fl_y_s))

    cv2.imshow('frame', numpy.vstack((stacked, stacked_mock)))

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    print("MSE -> ", ((stacked - stacked_mock)**2).mean(axis=None), 1/((time.perf_counter()-start_time)), end="\r")

# Destroy all the windows
cv2.destroyAllWindows()

mve_load.close()
