#!/usr/bin/python3
import time
import cv2
from display import Display
from extractor import Extractor
import numpy as np

W = 1920//2
H = 1080//2

F = 1
disp = Display(W, H)     # create a display object with specified width and height
K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])
fe = Extractor(K) # create an instance of the Extractor class


def process_frame(img):
    img = cv2.resize(img, (W, H))  # Resize the cv2 frame to the specified width and height
    matches = fe.extract(img)

    for pt1, pt2 in matches:
        u1, v1 = fe.denormalize(pt1) 
        u2, v2 = fe.denormalize(pt2)

        cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255,0,0))

    disp.paint(img) # Display the image with keypoints marked


if __name__ == "__main__":
    cap = cv2.VideoCapture("test.mp4") # Open the video file

    while cap.isOpened(): # Loop until the video is finished
        ret, frame = cap.read() # Read a frame from the video
        if ret:                         
            process_frame(frame)     # If the frame was successfully read, process it
        else:
            break # If the video reached the end (no frame was read), exit the loop
