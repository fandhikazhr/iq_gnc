#! /usr/bin/env python
# Import ROS.
import rospy
from mavros_msgs.srv import SetMode
from mavros_msgs.msg import State
# Import the API.
from iq_gnc.py_gnc_functions import *
# To print colours (optional).
from iq_gnc.PrintColours import *

def main():
    # Initializing ROS node.
    rospy.init_node("drone_controller", anonymous=True)

    # Create an object for the API.
    drone = gnc_api()
    # Wait for FCU connection.
    drone.wait4connect()
    # Wait for the mode to be switched.
    # drone.wait4start()
    
     # Set mode to GUIDED using MAVROS SetMode service
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

    # Create local reference frame.
    drone.initialize_local_frame()
    # Request takeoff with an altitude of 3m.
    drone.takeoff(1)
    # Specify control loop rate. We recommend a low frequency to not over load the FCU with messages. Too many messages will cause the drone to be sluggish.
    rate = rospy.Rate(5)

    # Specify some waypoints
    goals = [[0, 10, 3, 0], [0, 10, 0.5, 0], [0, 20, 3, 0],
             [0, 20, 0.5, 0], [0, 0, 3, 180], [0, 0, 3, 0]]
    #     goals = [[0, 0, 3, 0], [5, 0, 3, -90], [5, 5, 3, 0],
    #             [0, 5, 3, 90], [0, 0, 3, 180], [0, 0, 3, 0]]

    i = 0

    while i < len(goals):
        drone.set_destination(
            x=goals[i][0], y=goals[i][1], z=goals[i][2], psi=goals[i][3])
        rate.sleep()
        if drone.check_waypoint_reached():
            i += 1
    # Land after all waypoints is reached.
    drone.land()
    rospy.loginfo(CGREEN2 + "All waypoints reached landing now." + CEND)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()
