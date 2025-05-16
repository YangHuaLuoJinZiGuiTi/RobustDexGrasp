#!/usr/bin/env python3
import numpy as np
import rospy

from sensor_msgs.msg import JointState

import threading
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
#######################################################
"""This Controls the LEAP Hand and also sets up ros services that allow you to query the hand.

The services allow you to always have the latest data when you want it, and not spam the communication lines with unused data.

I recommend you only query using services when necessary and below 90 samples a second.  Each of position, velociy and current costs one sample, so you can sample all three at 30 hz or one at 90hz.

#Allegro hand conventions:
#0.0 is the all the way out beginning pose, and it goes positive as the fingers close more and more
#http://wiki.wonikrobotics.com/AllegroHandWiki/index.php/Joint_Zeros_and_Directions_Setup_Guide I belive the black and white figure (not blue motors) is the zero position, and the + is the correct way around.  LEAP Hand in my videos start at zero position and that looks like that figure.

#LEAP hand conventions:
#180 is flat out for the index, middle, ring, fingers, and positive is closing more and more.

Subscribes
----------
joint_angles

Services
----------
Makes joint angles, velocity and current available for reading.
Controls robotic hand
"""
########################################################
class LeapNode:
    def __init__(self):
        ####Some parameters to control the hand
        # self.ema_amount = float(rospy.get_param('/leaphand_node/ema', '1.0')) #take only current
        self.kP = float(rospy.get_param('/leaphand_node/kP', 800.0))
        self.kI = float(rospy.get_param('/leaphand_node/kI', 0.0))
        self.kD = float(rospy.get_param('/leaphand_node/kD', 200.0))
        self.curr_lim = float(rospy.get_param('/leaphand_node/curr_lim', 350.0)) #don't go past 600ma on this, or it'll overcurrent sometimes for regular, 350ma for lite.
        self.ema_amount = 0.2
        init_pose = np.array([0.5, 0., 0.3, 0.5, 0.5, 0., 0.3, 0.5, 0.5, 0., 0.3, 0.5, -1.5, 0.0, -0.1, 0.2])
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_urdf(init_pose)
        
        #subscribes to a variety of sources that can command the hand, and creates services that can give information about the hand out
        rospy.Subscriber("/leaphand_node/cmd_leap", JointState, self._receive_pose)
        rospy.Subscriber("/leaphand_node/cmd_allegro", JointState, self._receive_allegro)
        rospy.Subscriber("/leaphand_node/cmd_ones", JointState, self._receive_ones)
        
        self.joint_pub = rospy.Publisher('/leaphand_node/leap_joint', JointState, queue_size=1)
        self.pub_rate = rospy.Rate(500)  # 500Hz

        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB2', 4000000)
                self.dxl_client.connect()
        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # Dgain damping for side to side should be a bit less
        #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

        self.joint_state = JointState()
        self.joint_state.header.stamp = rospy.Time.now()
        self.joint_state.name = [f'leap_joint{i}' for i in range(16)]
        
        # 创建并启动发布线程
        self.pub_thread = threading.Thread(target=self.publish_joint_states)
        self.pub_thread.daemon = True  # 设置为守护线程，这样主程序退出时线程也会退出
        self.pub_thread.start()
        
    def publish_joint_states(self):
        while not rospy.is_shutdown():
            # 读取当前位置
            current_pos = self.dxl_client.read_pos()
            current_pos_allegro = lhu.urdf_to_allegro(current_pos) 
            self.joint_state.position = current_pos_allegro.tolist()
            # 发布消息
            self.joint_pub.publish(self.joint_state)
            rospy.sleep(0.1)


    #Receive LEAP pose and directly control the robot
    def _receive_pose(self, pose):
        pose = pose.position
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #Allegro compatibility, first read the allegro publisher and then convert to leap
    def _receive_allegro(self, pose):
        pose = lhu.allegro_to_urdf(pose.position)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #Sim compatibility, first read the sim publisher and then convert to leap
    def _receive_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose.position))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

#init the arm node
def main(**kwargs):
    rospy.init_node("leaphand_node")
    leaphand_node = LeapNode()
    while not rospy.is_shutdown():
        rospy.spin()

'''
Init node
'''
if __name__ == "__main__":
    main()
