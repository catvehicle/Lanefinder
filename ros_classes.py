import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from lanedetection.msg import Path
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class ros_setup:
    def __init__(self):
        rospy.init_node('lane_detection_node', anonymous=True)

class cmd_control_vel:
    def __init__(self, linear_vector, angular_vector):
        self.msg = Twist()
        self.msg.linear.x, self.msg.linear.y, self.msg.linear.z = linear_vector
        self.msg.angular.x, self.msg.angular.y, self.msg.angular.z = angular_vector

        self.publisher = rospy.Publisher('/cmd_control_vel', Twist, queue_size=10)
        #self.subscriber = rospy.Subscriber('/cmd_control_vel', Twist, self.callback)

    def publish(self):
        self.publisher.publish(self.msg)

class cmd_vel:
    def __init__(self, linear_vector, angular_vector):
        self.msg = Twist()
        self.msg.linear.x, self.msg.linear.y, self.msg.linear.z = linear_vector
        self.msg.angular.x, self.msg.angular.y, self.msg.angular.z = angular_vector

        self.publisher = rospy.Publisher('/catvehicle/cmd_vel', Twist, queue_size=10)
        #self.subscriber = rospy.Subscriber('/cmd_control_vel', Twist, self.callback)

    def publish(self):
        self.publisher.publish(self.msg)

class yaw_angle:
    def __init__(self, vect):
        self.msg = Twist()
        self.msg.linear.x, self.msg.linear.y, self.msg.linear.z = vect

        self.publisher = rospy.Publisher('/yaw_angle', Twist, queue_size=10)
        #self.subscriber = rospy.Subscriber('/yaw_angle', Twist, self.callback)

    def publish(self):
        self.publisher.publish(self.msg)

    def debug(self):
        x = self.msg.linear.x
        y = self.msg.linear.y
        z = self.msg.linear.z

        print("[{}, {}, {}]\n".format(x, y, z))

    def callback(self, msg):
        x = msg.linear.x
        y = msg.linear.y
        z = msg.linear.z

        rospy.loginfo("I heard %s %s %s\n", x, y, z)

class path:
    def __init__(self, xs, ys, phis):
        self.msg_x = Float32MultiArray()
	self.msg_y = Float32MultiArray()
	self.msg_phi = Float32MultiArray()

        self.msg_x.data = xs
        self.msg_y.data = ys
        self.msg_phi.data = phis

        self.publisher_x = rospy.Publisher('/xpath', Float32MultiArray, queue_size=10)
	self.publisher_y = rospy.Publisher('/ypath', Float32MultiArray, queue_size=10)
	self.publisher_phi = rospy.Publisher('/phipath', Float32MultiArray, queue_size=10)

    def publish(self):
        self.publisher_x.publish(self.msg_x)
	self.publisher_y.publish(self.msg_y)
	self.publisher_phi.publish(self.msg_phi)

    def debug(self):
        x = self.msg_x
        y = self.msg_y
        phi = self.msg_phi

        print("[{}, {}, {}]\n".format(x, y, phi))

class image_sub:
	def callback(self, data):
	    try:
            	self.cv2_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
	    except CvBridgeError, e:
            	print(e)

	def __init__(self):
	    self.bridge = CvBridge()
	    self.image_topic = "/triclops/left/image_color"
	    self.cv2_img = np.zeros((960,1280,3), np.uint8)

            rospy.Subscriber(self.image_topic, Image, self.callback)

	def getImg(self):
	    if np.array_equal(self.cv2_img, np.zeros((960,1280,3), np.uint8)):
		return None
	    else:
		return self.cv2_img






