#! /usr/bin/env python3
import rospy
import cv2
import torch
import sys
import time
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from mavros_msgs.srv import SetMode, CommandBool, CommandTOL
from iq_gnc.py_gnc_functions import *
from std_srvs.srv import Empty

# Tambahkan path YOLOv5 ke sys.path
yolov5_path = "yolov5_root_path"  # Sesuaikan dengan lokasi YOLOv5
sys.path.append(yolov5_path)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

class DroneYoloController:
    def __init__(self):
        rospy.init_node("drone_yolo_controller", anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = None  # Subscriber diaktifkan setelah takeoff
        self.vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)
        self.rate = rospy.Rate(10)

        # Load YOLOv5 model
        model_path = "yolov5_path_model/best.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DetectMultiBackend(model_path, device=self.device, dnn=False)
        self.model.eval()
        self.img_size = 640
        rospy.loginfo("YOLOv5 Model Loaded")

    def initialize_drone(self):
        rospy.wait_for_service('/mavros/set_mode')
        rospy.wait_for_service('/mavros/cmd/arming')
        rospy.wait_for_service('/mavros/cmd/takeoff')

        try:
            # Set mode ke GUIDED
            set_mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)
            response = set_mode_service(custom_mode="GUIDED")
            if response.mode_sent:
                rospy.loginfo("Mode set to GUIDED.")
            else:
                rospy.logwarn("Failed to set mode to GUIDED.")
            
            # Arming drone
            arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
            response = arm_service(True)  # Arming the drone
            if response.success:
                rospy.loginfo("Drone armed successfully.")
            else:
                rospy.logwarn("Failed to arm the drone.")

            # Takeoff ke ketinggian 1 meter
            takeoff_service = rospy.ServiceProxy('/mavros/cmd/takeoff', CommandTOL)
            response = takeoff_service(min_pitch=0, yaw=0, latitude=0, longitude=0, altitude=1)
            if response.success:
                rospy.loginfo("Takeoff successful")
            else:
                rospy.logwarn("Takeoff failed")

            # Tunggu beberapa detik agar drone stabil
            rospy.sleep(5)

            # Setelah takeoff berhasil, aktifkan deteksi YOLO
            self.image_sub = rospy.Subscriber("/webcam/image_raw", Image, self.image_callback)

        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            img = letterbox(cv_image, self.img_size, stride=32, auto=True)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR ke RGB, HWC ke CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device).float() / 255.0
            img = img.unsqueeze(0)
            
            with torch.no_grad():
                pred = self.model(img)
                pred = non_max_suppression(pred, 0.25, 0.45)
                
            detected = False
            object_center_x = 0
            for det in pred:
                if len(det):
                    detected = True
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], cv_image.shape).round()
                    
                    x1, y1, x2, y2 = det[0][:4]
                    object_center_x = (x1 + x2) / 2  
                    break
            
            self.control_drone(detected, object_center_x, cv_image.shape[1])
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def control_drone(self, detected, object_center_x, image_width):
        vel_msg = Twist()
        
        if detected:
            rospy.loginfo("Object detected, adjusting position.")
            offset = object_center_x - image_width / 2
            rospy.loginfo(f"Object center offset: {offset}")

            if abs(offset) > 20:  
                if offset > 0:  
                    rospy.loginfo("Objek Terlalu Ke Kanan, Nganan.")
                    vel_msg.linear.x = 0.2  
                    #vel_msg.angular.z = -0.1  
                if offset < 0:  
                    rospy.loginfo("Objek Terlalu Ke Kiri , Ngiri.")
                    vel_msg.linear.x = -0.2  
                    #vel_msg.angular.z = 0.1  
            if abs(offset) <= 20:
                rospy.loginfo("Objek sudah ditengah, bergerak maju.")
                vel_msg.linear.y = 0.5  
                #vel_msg.angular.z = 0.0  
        else:
            rospy.loginfo("Objek Tak Terlihat, Berhenti.")
            vel_msg.linear.x = 0.0
            vel_msg.linear.y = 0.0
            #vel_msg.angular.z = 0.0  
        
        self.vel_pub.publish(vel_msg)

if __name__ == '__main__':
    try:
        controller = DroneYoloController()
        controller.initialize_drone()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

