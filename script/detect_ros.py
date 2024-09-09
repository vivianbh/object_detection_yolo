#!/usr/bin/env python
import cv2
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import os
import sys
import numpy as np

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

class Detect():
    def __init__(self):
        rospy.init_node('detect_ros', anonymous=True)
        topic_name = "/uav0/camera_ir/camera/color/image_raw"
        self.img_sub = rospy.Subscriber(topic_name, Image, self.callback)
        self.img_pub = rospy.Publisher("/exp/camera/raw", Image, queue_size=10)
        self.img = Image()
        
        self.cv_img = np.ndarray(0)
        self.bridge = CvBridge()
        workingDir = os.getcwd()
        
        # Load the YOLOv8 model
        self.model = YOLO(workingDir+'/runs/detect/yolov8_trained_model4/weights/best.pt')

    def callback(self, msg):
        try:
            self.cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
    
    def image_publisher(self, frame):
        header = Header(stamp = rospy.Time.now())
        header.frame_id = "object"
        header = header
        self.img.height = 720
        self.img.width = 1280
        self.img.encoding = "bgr8"
        self.img.step = self.img.width*3
        self.img.data = np.array(frame).tobytes()
        self.img_pub.publish(self.img)

    def object_detect(self):
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = self.model.predict(self.cv_img)

        cls = results[0].boxes.cls
        conf = results[0].boxes.conf
        xywh = results[0].boxes.xywh
        speed = results[0].speed['preprocess']+results[0].speed['inference']+results[0].speed['postprocess']
        fps = 1000/speed

        print('result: ', cls, ' ', conf, ' ', xywh, ' ', speed, ' ', fps)
        print('xy: ', xywh.numpy())

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        self.image_publisher(annotated_frame)

        # Display the annotated frame
        resize = ResizeWithAspectRatio(annotated_frame, width=600)
        cv2.imshow("YOLO Prediction", resize)

if __name__ == "__main__":
    detect = Detect()
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        detect.object_detect()
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        rate.sleep()
    
    # Release the video capture object and close the display window
    cv2.destroyAllWindows()
