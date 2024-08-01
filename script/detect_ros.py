#!/usr/bin/env python
import cv2
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO

import sys
import numpy as np

import rospy
from sensor_msgs.msg import Image

class Detect():
    def __init__(self):
        rospy.init_node('detect_ros', anonymous=True)
        topic_name = "/uav0/camera_ir/camera/color/image_raw"
        self.img_sub = rospy.Subscriber(topic_name, Image, self.callback)
        
        self.cv_img = np.ndarray(0)
        self.bridge = CvBridge()
        
        # Load the YOLOv8 model
        self.model = YOLO("yolov8n.pt")

    def callback(self, msg):
        try:
            self.cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)

    def object_detect(self):
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = self.model.track(self.cv_img, persist=False)

        cls = results[0].boxes.cls
        conf = results[0].boxes.conf
        xywh = results[0].boxes.xywh
        speed = results[0].speed['preprocess']+results[0].speed['inference']+results[0].speed['postprocess']
        fps = 1000/speed

        print('result: ', cls, ' ', conf, ' ', xywh, ' ', speed, ' ', fps)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        #cv2.imshow("YOLOv8 Prediction", annotated_frame)

if __name__ == "__main__":
    detect = Detect()
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        detect.object_detect()
        
        # Break the loop if 'q' is pressed
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

        rate.sleep()
    
    # Release the video capture object and close the display window
    #cv2.destroyAllWindows()
