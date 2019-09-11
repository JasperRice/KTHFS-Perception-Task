'''
@author:        Sifan Jiang
email:          sifanj@kth.se
Date:           2019/9/10

Color Convention that we follow:
0 - YELLOW
1 - BLUE
2 - ORANGE
3 - WHITE
4 - BLACK
'''

import matplotlib.pyplot as plt
import numpy as np
import cv2  # Ubuntu
# from cv2 import cv2   # Windows

TESTMODE = True

def main():
    cap = cv2.VideoCapture('./Videos/Video_2.mp4')
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count = 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    orb = cv2.ORB_create()
    template_orange = cv2.imread('template_orange.png')
    kp_template, des_template = orb.detectAndCompute(template_orange, None)

    while count < frameCount:
        ret, frame = cap.read()
        bounding_box = []
        labels = []

        if ret:
            count += 1
            frame_cloned = np.copy(frame)

            # Covert BGR to Gray
            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Feature matching
            kp_frame, des_frame = orb.detectAndCompute(frame, None)
            matches = bf.match(des_template, des_frame)
            matches = sorted(matches, key=lambda x:x.distance)
            frame_matched = cv2.drawMatches(template_orange, kp_template,
                                            frame, kp_frame,
                                            matches[:50], None, flags=2)

            # cv2.imshow('Original frame', frame)
            # cv2.waitKey(10)
            # cv2.imshow('Result of cone detection', frame_cloned)
            # cv2.waitKey(10)

            if TESTMODE:
                cv2.imshow('Matching frame', frame_matched)
                cv2.waitKey(10)


if __name__ == '__main__':
    main()
