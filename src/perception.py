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

import numpy as np
import cv2  # Ubuntu
# from cv2 import cv2   # Windows

TESTMODE = True

def main():
    cap = cv2.VideoCapture('./Videos/Video_3.mp4')
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count = 0

    # hsv_yellow = np.array([30, 255, 255])
    lower_hsv_yellow = np.array([18, 63, 63])
    upper_hsv_yellow = np.array([35, 255, 255])

    # hsv_blue = np.array([120, 255, 255])
    lower_hsv_blue = np.array([109, 31, 31])
    upper_hsv_blue = np.array([140, 255, 255])

    # hsv_orange = np.array([19, 255, 255])MORPH_OPEN
    # hsv_red = np.array([0, 255, 255])
    lower_hsv_orange_a = np.array([0, 63, 100])
    upper_hsv_orange_a = np.array([15, 255, 255])

    lower_hsv_orange_b = np.array([159, 63, 63])
    upper_hsv_orange_b = np.array([179, 255, 255])

    # White
    lower_hsv_white = np.array([0, 0, 0])
    upper_hsv_white = np.array([179, 31, 255])

    # Black
    lower_hsv_black = np.array([0, 0, 0])
    upper_hsv_black = np.array([179, 255, 63])


    while count < frameCount:
        ret, frame = cap.read()
        bounding_box = []
        labels = []

        if ret:
            count += 1
            frame_cloned = np.copy(frame)
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            """ Color segmentation """
            frame_yellow = cv2.inRange(frame_hsv, lower_hsv_yellow, upper_hsv_yellow)
            frame_blue = cv2.inRange(frame_hsv, lower_hsv_blue, upper_hsv_blue)
            frame_orange = cv2.add(cv2.inRange(frame_hsv, lower_hsv_orange_a, upper_hsv_orange_a),
                                   cv2.inRange(frame_hsv, lower_hsv_orange_b, upper_hsv_orange_b))
            frame_black = cv2.inRange(frame_hsv, lower_hsv_black, upper_hsv_black)
            frame_white = cv2.inRange(frame_hsv, lower_hsv_white, upper_hsv_white)

            """ Erosion and dilation """
            kernel = np.ones((3,3), np.uint8)
            OPENING = True
            if OPENING:
                frame_yellow = cv2.morphologyEx(frame_yellow, cv2.MORPH_OPEN, kernel)
                frame_blue = cv2.morphologyEx(frame_blue, cv2.MORPH_OPEN, kernel)
                frame_orange = cv2.morphologyEx(frame_orange, cv2.MORPH_OPEN, kernel)
                frame_white = cv2.morphologyEx(frame_white, cv2.MORPH_OPEN, kernel)
                frame_black = cv2.morphologyEx(frame_black, cv2.MORPH_OPEN, kernel)
            else:
                frame_yellow = cv2.erode(frame_yellow, kernel, iterations=2)
                frame_yellow = cv2.dilate(frame_yellow, kernel, iterations=2)
                frame_blue = cv2.erode(frame_blue, kernel, iterations=2)
                frame_blue = cv2.dilate(frame_blue, kernel, iterations=2)
                frame_orange = cv2.erode(frame_orange, kernel, iterations=2)
                frame_orange = cv2.dilate(frame_orange, kernel, iterations=2)
                frame_white = cv2.erode(frame_white, kernel, iterations=2)
                frame_white = cv2.dilate(frame_white, kernel, iterations=2)
                frame_black = cv2.erode(frame_black, kernel, iterations=2)
                frame_black = cv2.dilate(frame_black, kernel, iterations=2)

            # frame_yellow = cv2.dilate(frame_yellow, kernel, iterations=5)
            # frame_yellow = cv2.erode(frame_yellow, kernel, iterations=5)
            frame_yellow = cv2.add(frame_yellow, frame_black)

            """ Smoothness """
            # frame_yellow = cv2.medianBlur(frame_yellow, 5)
            frame_yellow = cv2.GaussianBlur(frame_yellow, (5,5), 0)
            frame_yellow_smoothed = np.copy(frame_yellow)

            """ Edge detection """
            frame_yellow = cv2.Canny(frame_yellow, 100, 200)

            """ Contour approximation """
            _, contour_yellow, _ = cv2.findContours(np.array(frame_yellow), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # _, contour_yellow, _ = cv2.findContours(np.array(frame_yellow), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            approx_contour_yellow = []
            for c in contour_yellow:
                epsilon_c = 0.05 * cv2.arcLength(c, True)
                approx_contour_yellow.append(cv2.approxPolyDP(c, epsilon_c, closed=True))

            frame_yellow = np.zeros_like(frame_yellow)
            cv2.drawContours(frame_yellow, approx_contour_yellow, -1, (255, 255, 255), 1)

            """ Plot the bounding boxes """
            for box, i in zip(bounding_box, range(len(bounding_box))):
                xmin = box[0]
                ymin = box[1]
                xmax = box[2]
                ymax = box[3]
                # w = box[2]
                # h = box[3]

                if labels[i] == 0:
                    cv2.rectangle(frame_cloned ,(xmin,ymin), (xmax,ymax), (0,255,255), 5)
                    # cv2.rectangle(frame_cloned, (xmin,ymin), (xmin+w,ymin+h), (0,255,255), 5)
                    cv2.putText(frame_cloned, 'Y', (int(xmin),int(ymin)-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                if labels[i] == 1:
                    cv2.rectangle(frame_cloned ,(xmin,ymin), (xmax,ymax), (255,0,0), 5)
                    # cv2.rectangle(frame_cloned, (xmin,ymin), (xmin+w,ymin+h), (255,0,0), 5)
                    cv2.putText(frame_cloned, 'B', (int(xmin),int(ymin)-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                if labels[i] == 2:
                    cv2.rectangle(frame_cloned ,(xmin,ymin), (xmax,ymax), (0,165,255), 5)
                    # cv2.rectangle(frame_cloned ,(xmin,ymin), (xmin+w,ymin+h), (0,165,255), 5)
                    cv2.putText(frame_cloned, 'O', (int(xmin),int(ymin)-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # cv2.imshow('Original frame', frame)
            # cv2.waitKey(10)
            # cv2.imshow('Result of cone detection', frame_cloned)
            # cv2.waitKey(10)

            if TESTMODE:
                # cv2.imshow('HSV frame', frame_hsv)
                # cv2.waitKey(10)
                # cv2.imshow('Yellow smoothed', frame_yellow_smoothed)
                # cv2.waitKey(10)
                cv2.imshow('Yellow cones', frame_yellow)
                cv2.waitKey(10)
                # cv2.imshow('Blue cones', frame_blue)
                # cv2.waitKey(10)
                # cv2.imshow('Orange cones', frame_orange)
                # cv2.waitKey(10)


if __name__ == '__main__':
    main()
