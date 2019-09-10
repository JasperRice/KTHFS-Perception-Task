'''
@author:        Sifan Jiang
email:          sifanj@kth.se
Date:           2019/9/10   

Color Convention that we follow:
---------------- 
    0-  YELLOW
    1-  BLUE
    2-  ORANGE
    3-  WHITE
    4-  BLACK
'''

import numpy as np
from cv2 import cv2

def main():
    cap = cv2.VideoCapture('./Videos/Video_1.mp4')
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count = 0

    bounding_box = []
    labels = []
    # bounding_box.append(np.array((100,100,200,100)))
    # labels.append(0)

    # Read every frame till the end of video
    while count < frameCount:
        ret, frame = cap.read()
        if ret == True:
            count += 1
            frame_cloned = np.copy(frame)
            '''
            #
            #
            #
            #

            Your algorithm to process frame comes here
            The result may be:
            - a list of bounding_boxes (format corresponding to cv2.rectangle) and 
            - a list of labels (integer for simplicity)

            feel free to choose any other formulation of results
            #
            # 
            #
            # 
            '''
            for box, i in zip(bounding_box, range(len(bounding_box))):
                '''
                A quick note. A bounding box can have two formulations:
                1. xmin, ymin, w, h : which means first two numbers signify 
                the top left coordinate of rectangle and last two signify 
                its width and height respectively
                
                2. xmin, ymin, xmax, ymax : the first two coordinates signify
                the top left coordinate and last two coordinates signify the
                bottom right coordinate of rectangle.

                In our example, we use formulation 2, but its easy to interchange.
                follow comments.
                '''
                xmin = box[0]
                ymin = box[1]
                # xmax = box[2]
                w = box[2]
                # ymax = box[3]
                h = box[3]

                if labels[i] == 0:
                    # cv2.rectangle(frame_cloned ,(xmin,ymin), (xmax,ymax), (0, 255, 255), 5)
                    cv2.rectangle(frame_cloned, (xmin,ymin), (xmin+w,ymin+h), (0,255,255), 5)
                    cv2.putText(frame_cloned, 'Y', (int(xmin),int(ymin)-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                if labels[i] == 1:
                    # cv2.rectangle(frame_cloned ,(xmin, ymin), (xmax,ymax), (255, 0, 0), 5)
                    cv2.rectangle(frame_cloned, (xmin,ymin), (xmin+w,ymin+h), (255,0,0), 5)
                    cv2.putText(frame_cloned, 'B', (int(xmin),int(ymin)-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                if labels[i] == 2:
                    # cv2.rectangle(frame_cloned ,(xmin,ymin), (xmax,ymax), (0,165,255), 5)
                    cv2.rectangle(frame_cloned ,(xmin,ymin), (xmin+w,ymin+h), (0,165,255), 5)
                    cv2.putText(frame_cloned, 'O', (int(xmin),int(ymin)-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Original frame', frame)
            cv2.waitKey(10)
            cv2.imshow('Result of cone detection', frame_cloned)
            cv2.waitKey(10)


if __name__ == '__main__':
    main()