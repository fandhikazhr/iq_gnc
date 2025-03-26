#! /usr/bin/env python3
import rospy
import cv2
import torch
import sys
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from mavros_msgs.srv import SetMode
from iq_gnc.py_gnc_functions import *

# Tambahkan path YOLOv5 ke sys.path
yolov5_path = "/yolov5_root_path"  # Sesuaikan dengan lokasi YOLOv5
sys.path.append(yolov5_path)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

class DroneYoloController:
    def __init__(self):
        rospy.init_node("drone_yolo_controller", anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/webcam/image_raw", Image, self.image_callback)
        self.vel_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=10)  # Ganti topik di sini
        
        # Load YOLOv5 model
        model_path = "/model_yolov5_path/best.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DetectMultiBackend(model_path, device=self.device, dnn=False)
        self.model.eval()
        self.img_size = 640
        
        self.drone = gnc_api()
        self.drone.wait4connect()
        rospy.loginfo("YOLOv5 Model and Drone Controller Initialized")
        
    def initialize_drone(self):
        rospy.wait_for_service('/mavros/set_mode')
        try:
            set_mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)
            response = set_mode_service(custom_mode="GUIDED")
            if response.mode_sent:
                rospy.loginfo("Mode set to GUIDED.")
            else:
                rospy.logwarn("Failed to set mode to GUIDED.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
        self.drone.initialize_local_frame()
        self.drone.takeoff(3)
    
    def image_callback(self, msg):
        try:
            # Pastikan model sudah ada sebelum dipakai
            if not hasattr(self, 'model'):
               rospy.logwarn("YOLOv5 model belum siap, silahkan menunggu...")
               return
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
            object_center_x = 0  # Koordinat tengah objek
            for det in pred:
                if len(det):
                    detected = True
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], cv_image.shape).round()
                    
                    # Ambil koordinat tengah objek
                    x1, y1, x2, y2 = det[0][:4]
                    object_center_x = (x1 + x2) / 2  # Tengah objek (x)
                    break
            
            self.control_drone(detected, object_center_x, cv_image.shape[1])
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def control_drone(self, detected, object_center_x, image_width):
        vel_msg = Twist()
        
        if detected:
            rospy.loginfo("Object detected, adjusting position.")
            
            # Tentukan pergerakan drone berdasarkan posisi objek
            offset = object_center_x - image_width / 2  # Hitung offset dari tengah gambar
            rospy.loginfo(f"Object center offset: {offset}")

            # Jika objek tidak berada di tengah (lebih dari threshold 20 piksel)
            if abs(offset) > 20:  # Threshold untuk menggerakkan drone (dalam piksel)
                if offset > 0:  # Objek di kanan
                    rospy.loginfo("Object is on the right, moving drone left.")
                    vel_msg.linear.x = 0.1  # Gerakkan drone ke kiri
                if offset < 0:  # Objek di kiri
                    rospy.loginfo("Object is on the left, moving drone right.")
                    vel_msg.linear.x = -0.1  # Gerakkan drone ke kanan
            # Jika objek sudah cukup simetris (offset <= 20)
            if abs(offset) <= 20:
                rospy.loginfo("Object is centered, moving forward.")
                vel_msg.linear.y = 0.2  # Gerakkan maju jika objek di tengah
        else:
            rospy.loginfo("No object detected, stopping.")
            vel_msg.linear.x = 0.0
            vel_msg.linear.y = 0.0
        
        self.vel_pub.publish(vel_msg)

if __name__ == '__main__':
    try:
        controller = DroneYoloController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

