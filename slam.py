#!/usr/bin/python3
import time
import cv2
from display import Display
import numpy as np

# Specify the width and height for the display window, which is equal to 1/4 the resolution of the screen
W = 3024 // 4
H = 1964 // 4

disp = Display(W, H)     # create a display object with specified width and height


# Custom feature detector for ORB for more distributed features
class FeatureExtractor(object):
    def __init__(self):
        self.orb = cv2.ORB_create(100)

    def extract(self, img):

        feats = cv2.goodFeaturesToTrack(np.mean(img, axis = 2).astype(np.uint8), 3000, qualityLevel=0.099, minDistance=3)
        return feats


# create an instance of the FeatureExtractor class
fe = FeatureExtractor()


def process_frame(img):
    img = cv2.resize(img, (W, H))  # Resize the cv2 frame to the specified width and height
    kp = fe.extract(img)

    for f in kp: 
        u, v = map(lambda x: int(round(x)), f[0]) # Extract and round the position coordinate of each keypoint
        cv2.circle(img, (u,v), color=(0,255,0), radius = 3) # Draw a circle at each keypoint location
    disp.paint(img) # Display the image with keypoints marked


if __name__ == "__main__":
    cap = cv2.VideoCapture("test.mp4") # Open the video file

    while cap.isOpened(): # Loop until the video is finished
        ret, frame = cap.read() # Read a frame from the video
        if ret:                         
            process_frame(frame)     # If the frame was successfully read, process it
        else:
            break # If the video reached the end (no frame was read), exit the loop
