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
import cv2
# from cv2 import cv2

TESTMODE = True

def main():
    cap = cv2.VideoCapture('./Videos/Video_2.mp4')
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count = 0

    # hsv_yellow = np.array([30, 255, 255])
    lower_hsv_yellow = np.array([20, 63, 100])
    upper_hsv_yellow = np.array([40, 255, 255])

    # hsv_blue = np.array([120, 255, 255])
    lower_hsv_blue = np.array([110, 63, 100])
    upper_hsv_blue = np.array([130, 255, 255])

    # hsv_orange = np.array([19, 255, 255])
    # hsv_red = np.array([0, 255, 255])
    lower_hsv_orange = np.array([0, 63, 100])
    upper_hsv_orange = np.array([20, 255, 255])

    while count < frameCount:
        ret, frame = cap.read()
        bounding_box = []
        labels = []

        if ret:
            count += 1
            frame_cloned = np.copy(frame)

            ##### Test
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_yellow = cv2.inRange(frame_hsv, lower_hsv_yellow, upper_hsv_yellow)
            frame_blue = cv2.inRange(frame_hsv, lower_hsv_blue, upper_hsv_blue)
            frame_orange = cv2.inRange(frame_hsv, lower_hsv_orange, upper_hsv_orange)
            #####

            # Process cloned frame
            # bounding_box.append(np.array((100,100,200,100)))
            # labels.append(1)

            # Plot the bounding boxes
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

            cv2.imshow('Original frame', frame)
            cv2.waitKey(10)
            # cv2.imshow('Result of cone detection', frame_cloned)
            # cv2.waitKey(10)

            if TESTMODE:
                pass
                # cv2.imshow('HSV frame', frame_hsv)
                # cv2.waitKey(10)
                cv2.imshow('Yellow frame', frame_yellow)
                cv2.waitKey(10)
                cv2.imshow('Blue frame', frame_blue)
                cv2.waitKey(10)
                cv2.imshow('Orange frame', frame_orange)
                cv2.waitKey(10)


if __name__ == '__main__':
    main()
