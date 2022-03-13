# imports
import json
import sys
import argparse
import time
import imutils
import cv2
import datetime
import os
import signal
import numpy as np
from scripts import keyclipwriter

# importing the configuration from the config file
with open("config/mog.json", "r") as jsonfile:
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
parser.add_argument("--video",
                    type=str,
                    help="path to optional input video file")
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
    "CNT": cv2.bgsegm.createBackgroundSubtractorCNT(),
    "GMG": cv2.bgsegm.createBackgroundSubtractorGMG(),
    "MOG": cv2.bgsegm.createBackgroundSubtractorMOG(),
    "GSOC": cv2.bgsegm.createBackgroundSubtractorGSOC(),
    "LSBP": cv2.bgsegm.createBackgroundSubtractorLSBP()
}

# create the background substractors
fgbg = OPENCV_BG_SUBTRACTORS[conf["bug_sub"]]

# create the erosion and dilation kernels
ekernel = np.ones(tuple(conf["erode"]["kernel"]), "uint8")
dkernel = np.ones(tuple(conf["dilate"]["kernel"]), "uint8")

# initialize the key clip writer
# initialize the number of frame without motion
# initialize the number of frame since the last snpshot was written
kcw = keyclipwriter.KeyClipWriter(bufSize=conf["keyclipwriter_buffersize"])
frameWithoutMotion = 0
frameSinceSnap = 0

# capturing the interruption signal
signal.signal(signal.SIGINT, signal_handler)

# begining of the detection process
print("Detecting motion and storing videos..")
while True:
    # continious reading of frames
    fullframe = cap.read()
    # if not able to read the frame quit the program
    if fullframe is None:
        print("The streaming has ended.")
        break
    # get the frame from the tuple
    fullframe = fullframe[1]
    # start incrementing the frame counter since the first snap
    frameSinceSnap += 1
    # resize the frame
    frame = imutils.resize(fullframe, width=500)
    # apply the background substraction to the frame
    mask = fgbg.apply(frame)
    # clean up the mask with erodation and dilation
    mask = cv2.erode(mask, ekernel, iterations=conf["erode"]["iterations"])
    mask = cv2.dilate(mask, dkernel, iterations=conf["dilate"]["iterations"])
    # find all the contours in the mask
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # initialize a boolean variable to check if we,
    # frame this image or not
    motionThisFrame = False

    # looping over the contours to find the enclosing circle and rec
    for c in cnts:
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        (rx, ry, rw, rh) = cv2.boundingRect(c)
        (x, y, radius) = [int(v) for v in (x, y, radius)]
        # if the radius is not big enough we move to the next contour
        if radius < conf["min_radius"]:
            continue
        # if we reach this part then a motion has been detected
        print("Motion detected")
        # create a time stamp to name the image after it
        timestamp = datetime.datetime.now()
        timestring = timestamp.strftime("%Y%m%d-%H%M%S")
        # we set the boolean var to true since a motion has been detected
        motionThisFrame = True
        # we set this variable to zero to start the counting
        # in order to know when to stop recording
        frameWithoutMotion = 0
        # if annotation is set to true we draw circle and rec over the moving part
        if conf["annotate"]:
            cv2.circle(frame, (x, y), radius, (0, 0, 255), 2)
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        # if we are not already recording we set up path to the video,
        # and start the key event recorder
        if not kcw.recording:
            videoPath = os.path.sep.join((conf["output_path"], timestring))
            fourcc = cv2.VideoWriter_fourcc(*conf["codec"])
            kcw.start(f"{videoPath}.avi", fourcc, conf["fps"])

    # if there is no movement in the frame increment this variable
    if not motionThisFrame:
        frameWithoutMotion += 1

    # while the frames are coming update the frame buffer
    # and write to video if we are recording
    kcw.update(frame)

    # if we reach the threshold defined in conf we set this var to true
    noMotion = frameWithoutMotion >= conf["keyclipwriter_buffersize"]

    # if we are recording and there were no motion for x amount of frames
    # we stop the recording and release the recorder
    if kcw.recording and noMotion:
        kcw.finish()

    # display the video
    if conf["display"]:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
# if we go out of the reading frame loop we stop the recorder
if kcw.recording:
    kcw.finish()

# release the video streamer
cap.stop() if args.get("video", False) else cap.release()
