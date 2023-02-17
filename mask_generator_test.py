
# run rtsp
# https://github.com/aler9/rtsp-simple-server#from-a-webcam
# import the opencv library
import cv2
import numpy
from importables.video_decoder import VideoDecoder, DecodedVideoData
from importables.mask_generator import MaskGenerator, rev_letterbox
import time

# decoder = VideoDecoder("rtsp://192.168.0.101:8554/cam", 15)
decoder = VideoDecoder("/mnt/c/Skripsi/dataset-h264/R002A120/S018C001P008R002A120_rgb.mp4", 15)
yolo_maskgen = MaskGenerator(
    './libs/yolov7-mask/yolov7-mask.pt',
    './libs/yolov7-mask/data/hyp.scratch.mask.yaml',
    0.5,
    0.45)

print("start")

color = [numpy.random.randint(255), numpy.random.randint(255), numpy.random.randint(255)]
color = numpy.array(color, dtype=numpy.uint8)

while(True):

    # Capture the video frame
    # by frame
    start_time = time.perf_counter_ns()
    data = decoder.read()
    print(1000/((time.perf_counter_ns()-start_time) * 0.000001), end="\r")

    fr = data[1]
    # fr = cv2.flip(fr, 0)

    fr_letter = rev_letterbox(fr)

    mask_data = yolo_maskgen.generate([fr_letter])[0]

    mask_data_uint = (mask_data[1].astype(numpy.uint8)*255)

    mf = numpy.dstack((mask_data_uint, mask_data_uint, mask_data_uint))

    fmasked = numpy.copy(fr_letter)

    fmasked[mask_data[1]] = fmasked[mask_data[1]] * 0.5 + color * 0.5

    fshow = numpy.hstack((fr_letter, mf, fmasked))

    cv2.imshow('frame', fshow)

    # # the 'q' button is set as the
    # # quitting button you may use any
    # # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q') or not data[0]:
        break


# After the loop release the cap object
decoder.release()
# Destroy all the windows
cv2.destroyAllWindows()
