import numpy as np
import copy
from math import *

HIP_OFFSET = 0.0335
L1 = 0.08 # length of link 1
L2 = 0.11 # length of link 2

def calculate_forward_kinematics_robot(joint_angles: np.ndarray):
    """Calculate xyz coordinates of end-effector given joint angles.

    Use forward kinematics equations to calculate the xyz coordinates of the end-effector
    given some joint angles.

    Args:
      joint_angles: numpy array of 3 elements [TODO names]. Numpy array of 3 elements.
    Returns:
      xyz coordinates of the end-effector in the arm frame. Numpy array of 3 elements.
    """
    upperrot = joint_angles[2]
    lowerrot = joint_angles[1]
    baserot = -joint_angles[0]

    A0toEinA = np.array([0., 0., L2])

    matrixFromAtoB = np.array([
        [cos(upperrot), 0., -sin(upperrot)],
        [0., 1., 0.],
        [sin(upperrot), 0., cos(upperrot)],
    ])

    B0toA0inB = np.array([0., 0., L1])

    B0toEinB = B0toA0inB + np.matmul(matrixFromAtoB, A0toEinA)

    matrixfromBtoC = np.array([
        [cos(lowerrot), 0., -sin(lowerrot)],
        [0., 1., 0.],
        [sin(lowerrot), 0., cos(lowerrot)],
    ])

    C0toB0inC = np.array([0.,-HIP_OFFSET,0.])

    C0toEinC = C0toB0inC + np.matmul(matrixfromBtoC, B0toEinB)

    matrixfromCtoD = np.array([
        [cos(baserot), -sin(baserot), 0.],
        [sin(baserot), cos(baserot), 0.],
        [0., 0., 1.],
    ])

    C0toEinD = np.matmul(matrixfromCtoD, C0toEinC)
    D0toC0inD = np.array([0., 0., 0.])

    D0toEinD = D0toC0inD + C0toEinD

    return D0toEinD

def ik_cost(end_effector_pos: np.ndarray, guess: np.ndarray):
    """Calculates the inverse kinematics loss.

    Calculate the Euclidean distance between the desired end-effector position and
    the end-effector position resulting from the given 'guess' joint angles.

    Args:
      end_effector_pos: desired xyz coordinates of end-effector. Numpy array of 3 elements.
      guess: guess at joint angles to achieve desired end-effector position. Numpy array of 3 elements.
    Returns:
      Euclidean distance between end_effector_pos and guess. Returns float.
    """
    # TODO for students: Implement this function. ~1-5 lines of code.
    difference = end_effector_pos - calculate_forward_kinematics_robot(guess)
    return sqrt(np.linalg.norm(difference))

def calculate_jacobian(joint_angles):
    """Calculate the jacobian of the end-effector position wrt joint angles.
    
    Calculate the jacobian, which is a matrix of the partial derivatives
    of the forward kinematics with respect to the joint angles 
    arranged as such:
    
    dx/dtheta1 dx/dtheta2 dx/dtheta3
    dy/dtheta1 dy/dtheta2 dy/dtheta3
    dz/dtheta1 dz/dtheta2 dz/dtheta3
    
    Args:
      joint_angles: joint angles of robot arm. Numpy array of 3 elements.
    
    Returns:
      Jacobian matrix. Numpy 3x3 array.
    """

    def fdm(angles, angleindex, cartindex):
        deltascalar = .0001
        delta = np.array([0.,0.,0.])
        delta[angleindex] = deltascalar
        difference = calculate_forward_kinematics_robot(angles + delta) - calculate_forward_kinematics_robot(angles)
        return (difference/deltascalar)[cartindex]


    jacobian = np.array([
        np.array([fdm(joint_angles, angleindex, cartindex) for angleindex in range(3)])
        for cartindex in range(3)
    ])
    return jacobian

def calculate_inverse_kinematics(end_effector_pos, guess):
    """Calculates joint angles given desired xyz coordinates.

    Use gradient descent to minimize the inverse kinematics loss function. The
    joint angles that minimize the loss function are the joint angles that give 
    the smallest error from the actual resulting end-effector position to the
    desired end-effector position. You should use the jacobain function
    you wrote above.

    Args:
      end_effector_pos: Desired xyz coordinates of end-effector. Numpy array of 3 elements.
      guess: Guess at joint angles that achieve desired end-effector position. Numpy array of 3 elements.
    Returns:
      Joint angles that correspond to given desired end-effector position. Numpy array with 3 elements.
    """
    cost = ik_cost(end_effector_pos, guess)
    epsilon = .01
    learning_rate = 10.
    while (cost > epsilon):
        jacobian = calculate_jacobian(guess)
        difference = calculate_forward_kinematics_robot(guess) - end_effector_pos
        gradient = np.matmul(np.transpose(jacobian), difference)
        guess = guess - learning_rate*gradient
        cost = ik_cost(end_effector_pos, guess)
        print(f"cost: {cost}")
    return guess