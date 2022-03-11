import json
import sys
import argparse
import time
import imutils
import numpy as np
from scripts import keyclipwriter

# importing the configuration from the config file
with open("config/mog.json") as jsonfile:
    conf = json.load(jsonfile)

# define a function to handle the interruption signal
def signal_handler(sig, frame):
    print("You pressed 'ctrl+c'")
    print(f"Your file are saved in the {conf['output_path']} directory.")
    # checking if we are recording and wrap up
    if kcw.recording:
        kcw.finish()
    # exit the script
    sys.exit(0)

# define an argument parser for the script
parser = argparse.ArgumentParser()
parser.add_argument("--video",type=str,help="path to optional input video file")
args = vars(parser.parse_args())

# defining the video streamer
# if video is provided, read the video else stream from camera
if args["video"]:
    print(f"Opening video file '{args['video']}'")
    cap = cv2.VideoCapture(args["video"])
    time.sleep(3.0)
else:
    print("Starting video stream...")
    cap = cv2.VideoCapture(0)

# initialize the opencv background substractors
OPENCV_BG_SUBTRACTORS = {
    "CNT": cv2.bgsegm.createBackgroundSubtractorCNT,
    "GMG": cv2.bgsegm.createBackgroundSubtractorGMG,
    "MOG": cv2.bgsegm.createBackgroundSubtractorMOG,
    "GSOC": cv2.bgsegm.createBackgroundSubtractorGSOC,
    "LSBP": cv2.bgsegm.createBackgroundSubtractorLSBP
}

# create the background substractors
fgbg = OPENCV_BG_SUBTRACTORS[conf["bug_sub"]]

# create the erosion and dilation kernels
ekernel = np.ones(tuple(conf["erode"]["kernel"], "uint8"))
dkernel = np.ones(tuple(conf["dilate"]["kernel"], "uint8"))

# initialize the key clip writer
# initialize the number of frame without motion
# initialize the number of frame since the last snpshot was written
kcw = KeyClipWriter(bufSize=conf["keyclipwriter_buffersize"])
frameWithoutMotion = 0
frameSinceSnap = 0

# capturing the interruption signal
signal.signal(signal.SIGINT, signal_handler)

# begining of the detection process
print("Detecting motion and storing videos..")
while True:
    fullframe = cap.read()
    if fullframe is None:
        print("The streaming has ended.")
        break
    fullframe = fullframe[1] if args.get("video",False) else fullframe
    frameSinceSnap += 1
    frame = imutils.resize(fullframe,width=500)
    mask = fgbg.apply(frame)
    mask = cv2.erode(mask,ekernel,iterations=conf["erode"]["iterations"])
    mask = cv2.dilate(mask,dkernel,iterations=conf["dilate"]["iterations"])
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    motionThisFrame = False

    for c in cnts:
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        (rx, ry, rw, rh) = cv2.boundingRect(c)
        (x,y,radius) = [int(v) for v in (x,y,radius)]
        if radius < conf["min_radius"]:
            continue
        timestamp = datetime.datetime.now()
        timestring = timestamp.strftime("%Y%m%d-%H%M%S")
        motionThisFrame = True
        frameWithoutMotion = 0
        if conf["annotate"]:
            cv2.circle(frame, (x,y), radius, (0,0,255), 2)
            cv2.rectangle(frame, (rx,ry),(rx+rw,ry+rh),(0,255,0),2)


