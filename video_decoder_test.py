
# run rtsp
# https://github.com/aler9/rtsp-simple-server#from-a-webcam
# import the opencv library
import cv2
import numpy
from importables.video_decoder import VideoDecoder, DecodedVideoData
import time

# decoder = VideoDecoder("rtsp://192.168.0.101:8554/cam", 15)
decoder = VideoDecoder("/mnt/c/Skripsi/dataset-h264/R002A120/S018C001P008R002A120_rgb.mp4", 15)

while(True):

    start_time = time.perf_counter_ns()
    # Capture the video frame
    # by frame
    data = decoder.read()

    fr = cv2.flip(data[1], 0)
    fl_x = cv2.flip(data[2], 0)
    fl_y = cv2.flip(data[3], 0)

    # Display the resulting frame

    fl_x_s = numpy.dstack((fl_x, fl_x, fl_x))
    fl_y_s = numpy.dstack((fl_y, fl_y, fl_y))

    stacked = numpy.hstack((fr,
                            fl_x_s,
                            fl_y_s))

    stacked_rez = cv2.resize(stacked, (int(stacked.shape[1]/3), int(stacked.shape[0]/3)))

    cv2.imshow('frame', stacked_rez)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q') or not data[0]:
        break

    print(1000/((time.perf_counter_ns()-start_time) * 0.000001), end="\r")

# After the loop release the cap object
decoder.release()
# Destroy all the windows
cv2.destroyAllWindows()
