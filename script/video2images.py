import cv2
import os

cam = cv2.VideoCapture("video/output.mp4")

frame_rate = 2  # Desired frame rate (1 frame every 0.5 seconds)
frame_count = 1

while(True):
   ret,frame = cam.read()
   if ret:
      # if video is still left continue creating images
      name = 'images/' + str(frame_count) + '.jpg'

      if frame_count % int(cam.get(5)/frame_rate) == 0:
         cv2.imwrite(name, frame)
         print ('new frame captured...' + name)
      
      frame_count += 1
   else:
      break

cam.release()
cv2.destroyAllWindows()