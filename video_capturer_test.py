
# run rtsp
# https://github.com/aler9/rtsp-simple-server#from-a-webcam
# import the opencv library
import cv2
from importables.video_capturer import VideoCapturerProcessSpawner, FrameData
import time

webcam_ip = "rtsp://0.tcp.ap.ngrok.io:15225/cam"
video_path = "/mnt/c/Skripsi/dataset-h264/R002A120/S018C001P008R002A120_rgb.mp4"

# webcam: realtime normal fps
args = {
    "path": webcam_ip,
    "realtime": True,
    "update_rate": 60,
}
sampling_rate = 30

# webcam: not realtime, low sampling rate (should be lagging behind)
args = {
    "path": webcam_ip,
    "realtime": False,
    "update_rate": 60,
}
sampling_rate = 5

# webcam: not realtime, low sampling rate (should not be lagging behind)
args = {
    "path": webcam_ip,
    "realtime": True,
    "update_rate": 60,
}
sampling_rate = 5

# video: realtime high update rate will skip/lost frames ( >0% loss)
args = {
    "path": video_path,
    "realtime": True,
    "update_rate": 60,
}
sampling_rate = 30

# video: not realtime high update rate will not skip/lost frames ( =0% loss)
args = {
    "path": video_path,
    "realtime": False,
    "update_rate": 60,
}
sampling_rate = 30

# video: realtime high sampling rate will duplicate frames ( <0% loss)
args = {
    "path": video_path,
    "realtime": True,
    "update_rate": 30,
}
sampling_rate = 120

video_cap = VideoCapturerProcessSpawner(**args).start()
counter = 0


while(True):

    start_time = time.perf_counter()
    # Capture the video frame by frame
    data: FrameData = video_cap.read()

    counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q') or not data[0]:
        video_cap.stop()
        break

    cv2.imshow('frame', data[1])

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    time.sleep(1/sampling_rate)

    print(1/((time.perf_counter()-start_time)), end="\r")

# After the loop release the cap object
# decoder.stop()
# Destroy all the windows
cv2.destroyAllWindows()
print(counter, f"{round((100-counter)/100, 1)}% loss")
