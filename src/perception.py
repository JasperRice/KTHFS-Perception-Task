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

ADD = False

LOWER_CORNER = 3
UPPER_CORNER = 8
LOWER_RATIO = 0.55
UPPER_RATIO = 0.85
LOWER_WIDTH = 15
UPPER_WIDTH = 85
# LOWER_HEIGHT = 30
# UPPER_HEIGHT = 80
KERNEL_3x3 = np.ones((3,3), np.uint8)
KERNEL_5x5 = np.ones((5,5), np.uint8)
KERNEL_7x7 = np.ones((7,7), np.uint8)


def main():
    cap = cv2.VideoCapture('./Videos/Video_3.mp4')
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count = 0

    """ Yellow """
    lower_hsv_yellow = np.array([18, 63, 63])
    upper_hsv_yellow = np.array([35, 255, 255])

    """ Blue """
    lower_hsv_blue = np.array([109, 127, 20])
    upper_hsv_blue = np.array([140, 255, 255])

    """ Orange """
    lower_hsv_orange_a = np.array([0, 63, 63])
    upper_hsv_orange_a = np.array([10, 255, 255])

    lower_hsv_orange_b = np.array([169, 63, 63])
    upper_hsv_orange_b = np.array([179, 255, 255])

    """ White """
    lower_hsv_white = np.array([0, 0, 0])
    upper_hsv_white = np.array([179, 31, 255])

    """ Black """
    lower_hsv_black = np.array([0, 0, 0])
    upper_hsv_black = np.array([179, 255, 63])

    while count < frameCount:
        ret, frame = cap.read()

        if ret:
            count += 1
            frame_cloned = np.copy(frame)

            """ Convert BGR to HSV """
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            """ Color segmentation """
            frame_yellow = cv2.inRange(frame_hsv, lower_hsv_yellow, upper_hsv_yellow)
            frame_blue = cv2.inRange(frame_hsv, lower_hsv_blue, upper_hsv_blue)
            frame_orange = cv2.add(cv2.inRange(frame_hsv, lower_hsv_orange_a, upper_hsv_orange_a),
                                   cv2.inRange(frame_hsv, lower_hsv_orange_b, upper_hsv_orange_b))
            # frame_black = cv2.inRange(frame_hsv, lower_hsv_black, upper_hsv_black)
            # frame_white = cv2.inRange(frame_hsv, lower_hsv_white, upper_hsv_white)

            """ Erosion and dilation """
            # Opening
            frame_yellow = cv2.morphologyEx(frame_yellow, cv2.MORPH_OPEN, KERNEL_3x3)
            frame_blue = cv2.morphologyEx(frame_blue, cv2.MORPH_OPEN, KERNEL_5x5)
            frame_orange = cv2.morphologyEx(frame_orange, cv2.MORPH_OPEN, KERNEL_3x3)
            # frame_white = cv2.morphologyEx(frame_white, cv2.MORPH_OPEN, KERNEL_3x3)
            # frame_black = cv2.morphologyEx(frame_black, cv2.MORPH_OPEN, KERNEL_3x3)

            if ADD:
                frame_yellow = cv2.add(frame_yellow, frame_black)
            else:
                frame_yellow = cv2.dilate(frame_yellow, KERNEL_5x5, iterations=5)
                frame_yellow = cv2.erode(frame_yellow, KERNEL_5x5, iterations=5)
                # frame_blue = cv2.dilate(frame_blue, KERNEL_5x5, iterations=1)
                # frame_blue = cv2.erode(frame_blue, KERNEL_5x5, iterations=1)
                frame_orange = cv2.dilate(frame_orange, KERNEL_5x5, iterations=5)
                frame_orange = cv2.erode(frame_orange, KERNEL_5x5, iterations=5)

            """ Smoothness """
            frame_yellow = cv2.GaussianBlur(frame_yellow, (5,5), 0)
            frame_blue = cv2.GaussianBlur(frame_blue, (5,5), 0)
            frame_orange = cv2.GaussianBlur(frame_orange, (5,5), 0)

            """ Edge detection """
            frame_yellow = cv2.Canny(frame_yellow, 100, 200)
            frame_blue = cv2.Canny(frame_blue, 100, 200)
            frame_orange = cv2.Canny(frame_orange, 100, 200)

            frame_blue_cloned = np.copy(frame_blue)

            """ Contour approximation """
            _, contour_yellow, _ = cv2.findContours(np.array(frame_yellow), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            approx_contour_yellow = []
            for c in contour_yellow:
                epsilon_c = 0.05 * cv2.arcLength(c, True)
                approx_contour_yellow.append(cv2.approxPolyDP(c, epsilon_c, closed=True))

            _, contour_blue, _ = cv2.findContours(np.array(frame_blue), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            approx_contour_blue = []
            for c in contour_blue:
                epsilon_c = 0.05 * cv2.arcLength(c, True)
                approx_contour_blue.append(cv2.approxPolyDP(c, epsilon_c, closed=True))

            _, contour_orange, _ = cv2.findContours(np.array(frame_orange), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            approx_contour_orange = []
            for c in contour_orange:
                epsilon_c = 0.05 * cv2.arcLength(c, True)
                approx_contour_orange.append(cv2.approxPolyDP(c, epsilon_c, closed=True))

            """ Convex hull """
            convex_hull_yellow = []
            for ac in approx_contour_yellow:
                ch = cv2.convexHull(ac)
                if (LOWER_CORNER <= len(ch) <= UPPER_CORNER) and is_pointing_up(ch, frameHeight):
                    convex_hull_yellow.append(ch)

            convex_hull_blue = []
            for ac in approx_contour_blue:
                ch = cv2.convexHull(ac)
                if (LOWER_CORNER <= len(ch) <= UPPER_CORNER) and is_pointing_up(ch, frameHeight):
                    convex_hull_blue.append(ch)

            convex_hull_orange = []
            for ac in approx_contour_orange:
                ch = cv2.convexHull(ac)
                if (LOWER_CORNER <= len(ch) <= UPPER_CORNER) and is_pointing_up(ch, frameHeight):
                    convex_hull_orange.append(ch)

            """ Plot the bounding boxes """
            for ch in convex_hull_yellow:
                x,y,w,h = cv2.boundingRect(ch)
                cv2.rectangle(frame_cloned, (x,y), (x+w,y+h), (0,255,255), 5)
                cv2.putText(frame_cloned, 'Y', (int(x),int(y)-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            for ch in convex_hull_blue:
                x,y,w,h = cv2.boundingRect(ch)
                cv2.rectangle(frame_cloned, (x,y), (x+w,y+h), (255,0,0), 5)
                cv2.putText(frame_cloned, 'B', (int(x),int(y)-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            for ch in convex_hull_orange:
                x,y,w,h = cv2.boundingRect(ch)
                cv2.rectangle(frame_cloned, (x,y), (x+w,y+h), (0,165,255), 5)
                cv2.putText(frame_cloned, 'O', (int(x),int(y)-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # cv2.imshow('Original frame', frame)
            # cv2.waitKey(10)
            cv2.imshow('Result of cone detection', frame_cloned)
            cv2.waitKey(10)
            # cv2.imshow('Blue edge', frame_blue_cloned)
            # cv2.waitKey(10)


def is_pointing_up(cnt, frameHeight):
    points_above = []
    points_below = []
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h

    if ((LOWER_RATIO <= aspect_ratio <= UPPER_RATIO)
        and (LOWER_WIDTH <= w <= UPPER_WIDTH)
        and (y > 0.3 * frameHeight)):

        center = y + h/2
        for point in cnt:
            if point[0][1] < center:
                points_above.append(point)
            elif point[0][1] >= center:
                points_below.append(point)

        min_above = points_above[0][0][0]
        max_above = points_above[0][0][0]
        for point in points_above:
            if point[0][0] < min_above:
                min_above = point[0][0]
            if point[0][0] > max_above:
                max_above = point[0][0]

        min_below = points_below[0][0][0]
        max_below = points_below[0][0][0]
        for point in points_below:
            if point[0][0] < min_below:
                min_below = point[0][0]
            if point[0][0] > max_below:
                max_below = point[0][0]

        if (min_above > min_below) and (max_above < max_below):
            return True
    return False


if __name__ == '__main__':
    main()
