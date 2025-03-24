#!/usr/bin/env python3
import cv2
import rospy
import torch
import sys
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Tambahkan path YOLOv5 ke sys.path
yolov5_path = "yolov5_path_model"  # Sesuaikan dengan lokasi YOLOv5
sys.path.append(yolov5_path)

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
import numpy as np

class DroneYoloDetector:
    def __init__(self):
        # Inisialisasi node ROS
        rospy.init_node("drone_yolo_detector", anonymous=True)
        self.bridge = CvBridge()

        # Subscriber ke topik kamera
        self.image_sub = rospy.Subscriber("/webcam/image_raw", Image, self.image_callback)

        # Variabel kontrol untuk mengurangi lag
        self.last_time = rospy.Time.now()

        # Load model YOLOv5
        model_path = "/yolov5_path_model/best.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DetectMultiBackend(model_path, device=self.device, dnn=False)
        self.model.eval()  # Pastikan model dalam mode evaluasi
        
        self.img_size = 640  # Ukuran input YOLOv5

        rospy.loginfo("YOLOv5 Model loaded successfully!")

    def image_callback(self, msg):
        try:
            # Pastikan model sudah ada sebelum dipakai
            if not hasattr(self, 'model'):
               rospy.logwarn("YOLOv5 model belum siap, silahkan menunggu...")
               return
               
            current_time = rospy.Time.now()
            if (current_time - self.last_time).to_sec() < 0.033:  # Batasi FPS ke sekitar 30 FPS
                return
            self.last_time = current_time

            # Konversi dari ROS Image ke OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Preprocessing YOLOv5
            img = letterbox(cv_image, self.img_size, stride=32, auto=True)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR ke RGB, HWC ke CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0  # Normalisasi
            img = img.unsqueeze(0)  # Tambahkan dimensi batch

            # Prediksi YOLOv5
            with torch.no_grad():
                pred = self.model(img)
                pred = non_max_suppression(pred, 0.25, 0.45)

            # Proses hasil deteksi dan tampilkan di terminal
            detected_objects = []  # Untuk menyimpan hasil deteksi objek
            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], cv_image.shape).round()
                    for *xyxy, conf, cls in det:
                        label = f"{self.model.names[int(cls)]} {conf:.2f}"
                        detected_objects.append(label)  # Tambahkan label objek ke list

            # Jika ada objek terdeteksi, tampilkan di terminal
            if detected_objects:
                rospy.loginfo(f"Detected objects: {', '.join(detected_objects)}")
            else:
                rospy.loginfo("No objects detected.")

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    try:
        detector = DroneYoloDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

