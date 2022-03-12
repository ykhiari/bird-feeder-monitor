# imports
from collections import deque
from threading import Thread
from queue import Queue
import time
import cv2

class KeyClipWriter:

    def __init__(self, bufSize=64, timeout=1.0):
        # the max number of frames to be kept
        self.bufSize = bufSize
        # threading timeout value
        self.timeout = timeout
        # buffer of frames
        self.frames = deque(maxlen=bufSize)
        # queue of frames to be written to file
        self.Q = None
        # initialize the video writer
        self.writer = None
        # initialize a thread
        self.thread = None
        # boolean variable to start the recording
        self.recording = None

    def update(self, frame):
        # update the frame buffer
        self.frames.appendleft(frame)
        # if we are recording, update the queue as well
        if self.recording:
            self.Q.put(frame)

    def start(self, outputPath, fourcc, fps):
        # indicate the we are recoding, start the video writer,
        # and initialzie the queue of frames that need to be written
        # to the video file
        self.recording = True
        self.writer = cv2.VideoWriter(outputPath, fourcc, fps,
                             (self.frames[0].shape[1], self.frames[0].shape[0]), True)
        self.Q = Queue()

        # loop over the frames in the deque and put them in the queue
        for i in range(len(self.frames), 0, -1):
            self.Q.put(self.frames[i-1])
        
        # start a thread to write frame to video file
        self.thread = Thread(target=self.write, args=())
        self.thread.daemon = True
        self.thread.start()

    def write(self):
        while True:
            # if we are not recoding exit the thread
            if not self.recording:
                return
            
            # check to see if there are entries in the queue
            if not self.Q.empty():
                frame = self.Q.get()
                self.writer.write(frame)
            # otherwise, the queue is empty, so sleep for a bit
			# so we don't waste CPU cycles
            else:
                time.sleep(self.timeout)

    def flush(self):
        # empty the queue by flushing all remaining frames to file
        while not self.Q.empty():
            frame = self.Q.get()
            self.writer.write(frame)

    def finish(self):
        # indicate that we are done recording, join the thread,
		# flush all remaining frames in the queue to file, and
		# release the writer pointer
        self.recording = False
        if self.thread is not None:
            self.thread.join()
        self.flush()
        self.writer.release()
            
