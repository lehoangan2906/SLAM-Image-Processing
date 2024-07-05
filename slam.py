#!/usr/bin/python3
import time
import cv2
from display import Display

W = 3840 // 2
H = 2160 // 2

disp = Display(W, H)

def process_frame(img):
    img = cv2.resize(img, (W, H))  # Resize the frame
    disp.paint(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("test.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)
        else:
            break
