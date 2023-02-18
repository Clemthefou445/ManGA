from urdf_parser_py.urdf import URDF  # unified robot description format
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from pykdl_utils.kdl_kinematics import KDLKinematics
import PyKDL as _kdl  # kinematics and dynamics library
import rospy as _rospy  # quick interface for ROS topics, services and parameters
import numpy as np
import tf2_ros
import rospy
import torch
import random as rnd
import math

rospy.init_node("manipulability")  # initialize the ROS node for the process. Only one node per process


class PandaURDFModel():

    def __init__(self):
        self.baseLinkName = 'panda_link0'
        self.eeLinkName = 'palm'
        try:
            robot = URDF.from_parameter_server()

        except ConnectionRefusedError:
            print('Load Robot into parameter server first before accessing it!')
            raise (Exception)

        self.kdltree = kdl_tree_from_urdf_model(robot)
        self.ee_chain = self.kdltree.getChain(self.baseLinkName, self.eeLinkName)
        self.fk_ee = _kdl.ChainFkSolverPos_recursive(self.ee_chain)
        self.jointposition = _kdl.JntArray(7)  # we need to bring this to a torch tensor with differentiation
        # self.jointposition = torch.tensor(_kdl.JntArray(7), requires_grad= True) # we need to convert the kdlJntArray to torch tensor
        self.eeFrame = _kdl.Frame()
        self.jac_ee = _kdl.ChainJntToJacSolver(self.ee_chain)
        self.jacobian = _kdl.Jacobian(7)  # A jacobian instance is constructed
        # see .cpp : http://docs.ros.org/en/indigo/api/orocos_kdl/html/jacobian_8cpp_source.html#l00033

        # dynamics: (needs masses added to urdf!)
        self.grav_vector = _kdl.Vector(0., 0., -9.81)
        self.dynParam = _kdl.ChainDynParam(self.ee_chain, self.grav_vector)
        self.inertiaMatrix = _kdl.JntSpaceInertiaMatrix(7)

    def setJointPosition(self, jointPosition):
        for i in range(7):
            self.jointposition[i] = jointPosition[i]

    # self.jointposition[7] = jointPosition[7] # I don't get what use the gripper would have here

    def getEELocation(self):
        self.fk_ee.JntToCart(self.jointposition, self.eeFrame)
        return np.array(self.eeFrame.p), np.array(self.eeFrame.M)

    def getEEJacobian(self):
        self.jac_ee.JntToJac(self.jointposition, self.jacobian)
        # Here, JntToJac method is called on ChainJnttoJacSolver object and is given 2 arguments:
        # config vector and jacobian. Implementation of that method start line 48 in ChainJntToJacSolver.cpp

        # numpy array constructor does not work for kdl stuff.
        # There is likely to be a smarter way of doing this

        np_jac = np.zeros([6, 7])
        for row in range(6):
            for col in range(7):
                np_jac[row][col] = self.jacobian[row, col]
        return np_jac

    def getInertiaMatrix(self):
        self.dynParam.JntToMass(self.jointposition, self.inertiaMatrix)
        return self.inertiaMatrix


panda_kinematics = PandaURDFModel()  # plain instance creation
# Set initial joint position
joint_pos_initial = [0, 0, 0, 0, 0, 0, 0]  # radians
panda_kinematics.setJointPosition(joint_pos_initial)

tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

from scipy.spatial.transform import Rotation as R


def get_transformation(from_frame, to_frame):
    if tf_buffer._frameExists(from_frame) and tf_buffer._frameExists(to_frame):
        transform_base_EE = tf_buffer.lookup_transform(from_frame, to_frame, rospy.Time(0))
        # Convert quaternion into roation matrix
        quat_base_EE = transform_base_EE.transform.rotation
        r = R.from_quat([quat_base_EE.x, quat_base_EE.y, quat_base_EE.z, quat_base_EE.w])
        rot_matrix = r.as_matrix()
        # Create 4x4 transformation matrix
        trans_matrix = np.zeros((4, 4))
        trans_matrix[:3, :3] = rot_matrix
        trans_matrix[0, 3] = transform_base_EE.transform.translation.x
        trans_matrix[1, 3] = transform_base_EE.transform.translation.y
        trans_matrix[2, 3] = transform_base_EE.transform.translation.z
        # Return transformation matrix
        return trans_matrix
    else:
        raise Exception('Frame(s) are not available! - Check TF tree.')


t_base_EE = get_transformation("panda_link0", "panda_link8")

t_EE_palm = get_transformation("panda_link8", "palm")
t_palm_EE = get_transformation("palm", "panda_link8")

from franka_analytical_ik.srv import ik_request

rospy.wait_for_service('franka_ik_service')
ik_franka = rospy.ServiceProxy('franka_ik_service', ik_request)
q_actual = [-2, -0.044 + 0.3, 0.031, -2.427 + 0.4, -2.047, 0.845, -0.71]

panda_kinematics.getEELocation()

from differentiable_robot_model.robot_model import DifferentiableRobotModel as DRM
# instead of working with everything what's above, we work with a pytorch framework for kinematics
# That allows us to keep track of gradients at all times
import matplotlib.pyplot as plt
import random as rnd

robot = DRM(
    "/home/clementgillet/catkin_ws/src/franka_panda_description/robots/custom_panda_arm/modified_panda_arm.urdf")
# We use this because DRM doesn't know existence of palm
robotKDL = URDF.from_parameter_server()
kdl_kin = KDLKinematics(robotKDL, "panda_link0", "palm")

from rbohand3_kinematics.srv import fk2

# Connect to forward kinematics service
rospy.wait_for_service('forward_kinematics')
fk_rbo_hand = rospy.ServiceProxy('forward_kinematics', fk2)