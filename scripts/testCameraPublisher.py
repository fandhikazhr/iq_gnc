#!/usr/bin/env python3
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy

class DroneCameraViewer:
    def __init__(self):
        # Inisialisasi node
        rospy.init_node("drone_camera_viewer", anonymous=True)
        
        # Bridge untuk konversi ROS Image → OpenCV
        self.bridge = CvBridge()
        
        # Subscriber untuk topik kamera (Ganti sesuai topik kameramu)
        self.image_sub = rospy.Subscriber("/webcam/image_raw", Image, self.image_callback)
        
    def image_callback(self, msg):
        try:
            # Konversi dari ROS Image ke OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Tampilkan gambar dalam jendela
            cv2.imshow("Drone Camera Feed", cv_image)
            cv2.waitKey(1)  # Update tampilan
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    try:
        # Jalankan visualisasi kamera
        cam_viewer = DroneCameraViewer()
        rospy.spin()  # Biarkan node terus berjalan
    except rospy.ROSInterruptException:
        cv2.destroyAllWindows()

