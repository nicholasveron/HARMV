
# run rtsp
# https://github.com/aler9/rtsp-simple-server#from-a-webcam
# import the opencv library
import cv2
import numpy
from importables.optical_flow_generator import OpticalFlowGenerator
from importables.motion_vector_extractor import MotionVectorExtractorProcessSpawner
import time

# decoder = MotionVectorExtractorProcessSpawner("rtsp://192.168.0.186:8554/cam", 15, 60, 120, True, 320, box=True).start()
decoder = MotionVectorExtractorProcessSpawner("rtsp://0.tcp.ap.ngrok.io:15225/cam", 15, 30, 30, True, 320).start()
opgen = OpticalFlowGenerator("raft_small", "things", 15)
# opgen = OpticalFlowGenerator("flowformer", "chairs", 15)
# decoder = MotionVectorExtractorProcessSpawner("/mnt/c/Skripsi/dataset-h264/R001A001/S001C001P001R001A001_rgb.mp4", 15, 5, 5, True, 320).start()
# decoder = MotionVectorExtractorProcessSpawner("/mnt/c/Skripsi/dataset-h264-libx/R001A001/S001C001P001R001A001_rgb.mp4", 15, 10, 10, True, 320).start()
last_fr = numpy.empty((1))
while(True):

    start_time = time.perf_counter()
    # Capture the video frame by frame
    data = decoder.read()

    if cv2.waitKey(1) & 0xFF == ord('q') or not data[0]:
        decoder.stop()
        break

    fr = data[1]

    if len(last_fr.shape) == 1:
        last_fr = fr.copy()

    op_fl = opgen.generate_once(last_fr, fr)

    last_fr = fr.copy()

    fl_x = op_fl[0]
    fl_y = op_fl[1]

    # fl_x = data[2]
    # fl_y = data[3]

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
