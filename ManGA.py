# Import of useful kinematics- and ros-related libraries are done in kdl_Utils
from kdl_Utils import *

import torch
import numpy as np

import time
from tkinter import *

# Global variable; If True, the Joint Limit Penalty is implemented (you can change it in the argument of gradientAscent())
JLP = False


# Compute the nullspace projection of the jacobian for any initial pose vector
# This function is called at every gradient step so the locality of the Jacobian is accounted for
def nullSpace_Proj(joint_init, constrainedDimensions):
    J_linear, J_angular = robot.compute_endeffector_jacobian(joint_init, "panda_link8")
    torchJ = torch.cat((J_linear, J_angular))
    # Remove the rows of the jacobian that are not constrained
    list_to_tensor = torch.t(torch.unsqueeze(torch.Tensor(constrainedDimensions), 0))
    comp_jac = torch.cat((list_to_tensor, torchJ), 1)
    filtered = comp_jac[comp_jac[:, 0] == 1]
    constr = filtered[:, 1:]

    # Apply the definition of nullspace projection (moore-penrose pseudo-inverse is used)
    null_space_proj = torch.eye(9) - torch.matmul(torch.linalg.pinv(constr), constr)
    # Apply the definition of nullspace projection (pseudo inverse full formula)
    # null_space_proj = torch.eye(9) - torch.matmul(torch.transpose(constr,0,1), torch.matmul(torch.linalg.inv(torch.matmul(constr, torch.transpose(constr,0,1))), constr))
    return null_space_proj


# Penalize the manipulability objective function for being too close to the joint limits
# When the global variable JLP (above) is True, the result of this function is multiplied with the original manipulability
# Formula was borrowed from (Vahrenkamp, 2012)
def jointLimitPenalty(joint_init, qDelta, null_coeff, k):
    # k is a significant hyperparameter of the model
    result = 1
    for j in range(7):
        numerator = ((joint_init[j] + torch.matmul(null_coeff, qDelta))[j] - robot.get_joint_limits()[j]['lower']) * (
                    robot.get_joint_limits()[j]['upper'] - (joint_init[j] + torch.matmul(null_coeff, qDelta)[j]))
        denominator = (robot.get_joint_limits()[j]['upper'] - robot.get_joint_limits()[j]['lower']) ** 2
        result *= numerator / denominator
    inExp = (-k) * (result)
    j = 1 - torch.exp(inExp)
    return j


# Computes the main manipulability measure in function of the current joint configuration
# Formula was partially borrowed from (Yoshikawa, 1985)
def manipulability(joint_init, qDelta, constrainedDimensions, null_coeff, k):
    newPose = joint_init + torch.matmul(null_coeff, qDelta)
    J_linear, J_angular = robot.compute_endeffector_jacobian(newPose, "panda_link8")
    torchJ = torch.cat((J_linear, J_angular))
    torchJT = torch.transpose(torchJ, 0, 1)

    JJT = torch.matmul(torchJ, torchJT)
    det = torch.linalg.det(JJT)
    w = torch.sqrt(det)
    # If True,  Joint Limit Penalty is integrated
    if JLP:
        return w * jointLimitPenalty(joint_init, qDelta, null_coeff, k)
    return w


# At every Gradient Step, this function visually outputs the current joint configuration into the RViz Simulator
def simulatorView(config):
    poses_fk = fk_rbo_hand(index_airmass=[0, 0],
                           middle_airmass=[0, 0],
                           ring_airmass=[0, 0],
                           little_airmass=[0, 0],
                           thumb_airmass=[0, 0.0, 0.0, 0.0],
                           palm_airmass=[0.],
                           in_palm_frame=1,
                           scaled_masses=1,
                           panda_joint_angles=config[:7])


# A random initial pose is returned
# The output depends on the user preferences
# If a well-defined 7D q-vector is given, than the Gradient Ascent (GA) will start with this given configuration.
# If nothing is given, the GA starts with an unconstrained, arbitrary pose in space.
# If an initial pose constraint is given (like "upwards"), the  GA starts with a randomly initialized pose within the constraint.
def randomPoseInitializer(desiredPose, constrainedDimensions):
    q = randomConfig()

    if type(desiredPose) == list:
        joint_init = torch.cat(
            (torch.tensor(desiredPose, requires_grad=True), torch.tensor([0., 0.], requires_grad=True)))
        return joint_init

    elif type(desiredPose) == None:
        joint_init_random = torch.cat((torch.tensor(q, requires_grad=True), torch.tensor([0., 0.], requires_grad=True)))
        return joint_init_random

    elif type(desiredPose) == str:
        robotKDL = URDF.from_parameter_server()
        kdl_kin = KDLKinematics(robotKDL, "panda_link0", "palm")
        if desiredPose == "upwards":
            motherQ = [-2.0, 0.0, 2.3259260654449463, -2.4263031482696533, -2.0412862300872803, 0.8505620360374451,
                       -0.8299999833106995]
        else:
            motherQ = [-2.497678279876709, 1.2272013425827026, 2.727571487426758, -2.5281753540039062,
                       0.3514966368675232, 1.649809718132019, -0.8299999833106995]
        t_base_Palm = kdl_kin.forward(motherQ)

        isNan = True
        # Here, we try to find a solution with the IK-solver ik_franka() , while there is no solution, we keep trying random rot and trans
        while isNan:
            rot = randomRotation(desiredPose, t_base_Palm)
            rotTrans = randomTranslation(rot, q)

            temp = ik_franka(
                # indicates the pose of the EE to calculate IK from
                desired_transform=rotTrans,
                q7=-0.83,  # q0,q1,q2,q3,q4,q5,q6 --> 7-dim Vector ;;;; q7 is joint 8 --> wrist
                q_actual=motherQ)

            temp = [temp.solution_1, temp.solution_2, temp.solution_3, temp.solution_4]

            for solu in temp:
                if not math.isnan(solu[0]):
                    # Show the random initialized configuration in the simulator
                    simulatorView(solu)
                    isNan = False
                    joint_init = torch.cat(
                        (torch.tensor(solu, requires_grad=True), torch.tensor([0., 0.], requires_grad=True)))
                    return joint_init
            q = randomConfig()


# This function outputs a unconstrained, random initial pose
def randomConfig():
    q = []
    for j in range(7):
        randAngle = rnd.uniform(robot.get_joint_limits()[j]['lower'], robot.get_joint_limits()[j]['upper'])
        q.append(randAngle)
    return q


# Here, we apply a random rotation around z
def randomRotation(desiredPose, HT):
    randAngle = rnd.uniform(-360, 360)

    rotationZ = np.array(
        [(math.cos(randAngle), -math.sin(randAngle), 0, 0), (math.sin(randAngle), math.cos(randAngle), 0, 0),
         (0, 0, 1, 0), (0, 0, 0, 1)])
    t_base_Palm = np.matmul(HT, rotationZ)
    conversion = get_transformation("palm", "panda_link8")
    t_base_EE = np.matmul(t_base_Palm, conversion)
    return t_base_EE


# Here, we apply a random translation
# More precisely, the function takes a random, unconstrained q
# Afterwards, the q is translated to a Homogeneous Transform and the translational part is retrieved from that HT
# We apply that arbitrarily chosen translation to our already randomly rotated pose by inserting it in the new HT
def randomTranslation(result, q):
    kdl_kin_EE = KDLKinematics(robotKDL, "panda_link0", "panda_link8")
    rand_t_base_Palm = kdl_kin_EE.forward(q)

    result[:3, 3] = rand_t_base_Palm[:3, 3]
    desired = result.transpose().reshape(16).tolist()
    desiredFinal = []
    for item in desired[0]:
        desiredFinal.append(float(item))
    return desiredFinal


# Here is the main function of the program, that is being called externally.
# All parameters are initialized, and the function can be called without parameters as gradientAscent()
def gradientAscent(initialPose=None, constrainedDimensions=[0, 0, 0, 0, 0, 0], lr=8, c=0.00001, P=False, k=5000):
    global JLP
    # The global variable takes the boolean value that from user input
    JLP = P
    # Find a random initial pose according to the input constraint
    joint_init = randomPoseInitializer(initialPose, constrainedDimensions)
    # Bring the franka emika in a random initial pose in the simulator (constrained or not)
    simulatorView(joint_init)

    # Gives the user 3 seconds to watch the initial pose before the optimization procedure is kickstarted
    time.sleep(3)

    # The configuration delta is initialized at zero and will be iterated on to progress towards an optimal configuration
    qDelta = torch.zeros(9, requires_grad=True)
    # Compute the nullspace projection matrix
    null_coeff = nullSpace_Proj(joint_init, constrainedDimensions)

    # Sidetrick to make the convergence condition work
    w_array = [0, 1]

    # Plain Vanilla Gradient Ascent with an early stopping "while condition" at convergence
    # early stopping treshold is also an important hp of the system

    while abs(w_array[-1] - w_array[-2]) > c:
        # Compute the MAIN OBJECTIVE FUNCTION (with or without JLP)
        w = manipulability(joint_init, qDelta, constrainedDimensions, null_coeff, k)
        w_array.append(w)
        # Differientiate on the main objective function (with or without JLP)
        w.backward(retain_graph=True)
        # Take the GRADIENT OF QDELTA (the variable on which the GA is performed)
        grad = qDelta.grad
        update = grad * lr  # learning rate for adapting qDelta at each iteration (important hyperparameter of the system)
        # GRADIENT STEP
        qDelta = torch.tensor((qDelta + update), requires_grad=True)
        # Update the current pose of the system
        pose = joint_init + torch.matmul(null_coeff, qDelta)

        # Account for locality of Jacobian by reparametrizing joint_init and nullspace
        joint_init = pose
        qDelta = torch.zeros(9, requires_grad=True)
        null_coeff = nullSpace_Proj(joint_init, constrainedDimensions)
        simulatorView(pose[:7])
        
    return [w_array[2:], joint_init]


def forwardKinematics(q):
    return robot.compute_forward_kinematics(q, "panda_link8")

