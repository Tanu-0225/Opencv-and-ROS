#!/usr/bin/env python3
import rospy 
from sensor_msgs.msg import Image 
from std_msgs.msg import String
from cv_bridge import CvBridge 
import cv2
from cv2 import *
import numpy as np
import image_slicer

key_mapping = ['T','R','B','G','W','S','Z']

def key_node(msg):
    global key
    if len(msg.data) == 0 or msg.data not in key_mapping:
        return
    key = 'n'
    key = str(msg.data)
"""
    if key =='T':
        print("Input Letter:",key)
        cv2.destroyWindow("camera")
        cv2.waitKey(0)
        #gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)]
        #cv2.imshow("camera", current_frame)
	cv2.imshow("gray", gray)
	cv2.waitKey(1)
"""
 
def callback(data):
 
  # Used to convert between ROS and OpenCV images
  br = CvBridge()
  rospy.loginfo("receiving video frame")
  # Convert ROS Image message to OpenCV image
  current_frame = br.imgmsg_to_cv2(data)
  #Grayscale
  gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
  #Threshold
  threshVal = 75
  ret,thresh = cv2.threshold(gray, threshVal, 255, cv2.THRESH_BINARY)
  drawImg = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
  hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
  
  # Set range for red color 
  red_lower = np.array([136, 87, 111], np.uint8)
  red_upper = np.array([180, 255, 255], np.uint8)
  red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
  # Set range for green color 
  green_lower = np.array([25, 52, 72], np.uint8)
  green_upper = np.array([102, 255, 255], np.uint8)
  green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
  # Set range for blue color 
  blue_lower = np.array([94, 80, 2], np.uint8)
  blue_upper = np.array([120, 255, 255], np.uint8)
  blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
  # For red color
  red_mask = cv2.dilate(red_mask, kernal)
  res_red = cv2.bitwise_and(imageFrame, imageFrame, 
                              mask = red_mask)
  # For green color
  green_mask = cv2.dilate(green_mask, kernal)
  res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask = green_mask)
  # For blue color
  blue_mask = cv2.dilate(blue_mask, kernal)
  res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                               mask = blue_mask)
                               
  #Split image into two halves
  cam_port = 0
  cam = VideoCapture(cam_port)
  result, image = cam.read()
  image_slicer.slice('image',2)
  
  # Display image
  cv2.imshow("camera", current_frame)
  cv2.waitKey(1)
  fkey = str(key)
	
  if fkey==key_mapping[0]:
      cv2.destroyAllWindows()
      cv2.imshow("THRESHOLD", drawImg)
      cv2.waitKey(1)
		
  elif fkey==key_mapping[1]:
      cv2.destroyAllWindows()
      cv2.imshow("Red", res_red)
      cv2.waitKey(1)
		
  elif fkey==key_mapping[2]:
      cv2.destroyAllWindows()
      cv2.imshow("Blue", res_blue)
      cv2.waitKey(1)
  elif fkey==key_mapping[3]:
      cv2.destroyAllWindows()
      cv2.imshow("Green", res_green)
      cv2.waitKey(1)
  elif fkey==key_mapping[4]:
      cv2.destroyAllWindows()
      cv2.imshow("BlackandWhite", gray)
      cv2.waitKey(1)
  elif fkey==key_mapping[5]:
      cv2.destroyAllWindows()
      cv2.imshow("Split", image)
      cv2.waitKey(1)
		
		
  elif fkey=='n':
      cv2.destroyAllWindows()
      cv2.imshow("camera", current_frame)
      cv2.waitKey(1)
	
  elif fkey==key_mapping[6]:
      cv2.destroyAllWindows()
      
      
def receive_message():
 
  # Tells rospy the name of the node.
  # Anonymous = True makes sure the node has a unique name. Random
  # numbers are added to the end of the name. 
  rospy.init_node('video_sub_py', anonymous=True)
   
  # Node is subscribing to the video_frames topic
  rospy.Subscriber('video_frames', Image, callback)
 
  # spin() simply keeps python from exiting until this node is stopped
  rospy.spin()
 
  # Close down the video stream when done
  cv2.destroyAllWindows()
  
if __name__ == '__main__':
  receive_message()
