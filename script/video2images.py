import cv2
import os

cam = cv2.VideoCapture("video/output.mp4")

frameno = 1
while(True):
   ret,frame = cam.read()
   if ret:
      # if video is still left continue creating images
      name = 'images/' + str(frameno) + '.jpg'
      print ('new frame captured...' + name)

      cv2.imwrite(name, frame)
      frameno += 1
   else:
      break

cam.release()
cv2.destroyAllWindows()