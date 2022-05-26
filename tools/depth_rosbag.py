# Python 2.7
import rospy
import message_filters
from nav_msgs.msg import Odometry
import csv


class Nodo(object):
    def __init__(self):
        self.drone_odom_position = None
        self.drone_odom_orientation = None
        drone_sub = message_filters.Subscriber("/moving_object_odom", Odometry)
        opel_sub = message_filters.Subscriber("/opel/odom", Odometry)
        ts = message_filters.TimeSynchronizer([drone_sub, opel_sub], 60)#, 0.2,allow_headerless=False)
        ts.registerCallback(self.callback)
        
    def callback(self, drone_odom, opel_odom):
        # global drone_odom_position, drone_odom_orientation
        self.drone_odom_position = drone_odom.pose.pose.position
        self.drone_odom_orientation = drone_odom.pose.pose.orientation
        self.opel_odom_position = opel_odom.pose.pose.position
        self.opel_odom_orientation = opel_odom.pose.pose.orientation
        
        with open(r'depth_data.csv', 'a') as csvfile:
            fieldnames = ['x_drone','y_drone', 'z_drone', 'x_opel', 'y_opel', 'z_opel']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'x_drone': self.drone_odom_position.x, 'y_drone':self.drone_odom_position.y, 'z_drone': self.drone_odom_position.z,
                             'x_opel': self.opel_odom_position.x, 'y_opel':self.opel_odom_position.y, 'z_opel': self.opel_odom_position.z})
        

if __name__ == "__main__":
    rospy.init_node("data_collector", anonymous=True)

    my_node = Nodo()
    rospy.spin()