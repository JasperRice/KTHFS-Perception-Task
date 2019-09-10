'''
Change the fields below for your code to make it more presentable:

@author :       Ajinkya Khoche
email:          khoche@kth.se
Date:           2018/09/03   
Description:    This program
                - Reads a video from 'test_videos' folder and reads every frame 
                till end of video. It stores every frame in variable of same name.
                - Your algorithm should process 'frame' variable (or frame_cloned. 
                its good to clone the frame to preserve original data)
                - The result of your algorithm should be lists of 'bounding_boxes'
                and 'labels'. 
                - The helper code takes 'bounding_boxes' to draw rectangles on the
                positions where you found the cones. It uses corresponding 'labels'
                to name which type of cone was found within 'bounding_boxes'.  

                Color Convention that we follow:
                ---------------- 
                    0-  YELLOW
                    1-  BLUE
                    2-  ORANGE
                    3-  WHITE
                    4-  BLACK

                This basically means that if labels[i] = 0, then you can set the i_th
                bounding_box as 'yellow cone'    
'''
import numpy as np
from cv2 import cv2

def main():
    # Read video from disk and count frames
    cap = cv2.VideoCapture('./Videos/Formula Student Spain 2015 Endurance- DHBW Engineering with the eSleek15.mp4')
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    count = 0
    
    '''
    NOTE: For this example, bounding_box is manually defined.
    For your assignment you need to comment the definitions of
    'bounding_boxes' and 'labels' below and use bounding_box 
    obtained from your algorithm. Its the structure which is 
    provided
    '''
    bounding_box = []
    bounding_box.append(np.array((100,100,150,200)))
    bounding_box.append(np.array((200,200,300,400)))
    bounding_box.append(np.array((500,500,600,700)))
    
    labels = [0,2,1]

    # Read every frame till the end of video
    while count < frameCount:
        ret, frame = cap.read()
        if ret == True:
            count = count + 1

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
                xmax = box[2]   # w = box[2]
                ymax = box[3]   # h = box[3]

                if labels[i] == 0:
                    cv2.rectangle(frame_cloned ,(xmin,ymin), (xmax,ymax), (0, 255, 255), 5)     #cv2.rectangle(frame_cloned ,(xmin,ymin), (xmin + w,ymin + h), (0,255,0), 5)
                    cv2.putText(frame_cloned, 'yellow cone', (int(xmin),int(ymin)-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                if labels[i] == 1:
                    cv2.rectangle(frame_cloned ,(xmin,ymin), (xmax,ymax), (255, 0, 0), 5)     #cv2.rectangle(frame_cloned ,(xmin,ymin), (xmin + w,ymin + h), (0,255,0), 5)
                    cv2.putText(frame_cloned, 'blue cone', (int(xmin),int(ymin)-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                if labels[i] == 2:
                    cv2.rectangle(frame_cloned ,(xmin,ymin), (xmax,ymax), (0,165,255), 5)     #cv2.rectangle(frame_cloned ,(xmin,ymin), (xmin + w,ymin + h), (0,255,0), 5)
                    cv2.putText(frame_cloned, 'orange cone', (int(xmin),int(ymin)-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.imshow('Original frame', frame)
            cv2.waitKey(10)
            cv2.imshow('Result of cone detection', frame_cloned)
            cv2.waitKey(10)
if __name__ == '__main__':
    main()