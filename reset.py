import time

import numpy as np
from pupper_hardware_interface import interface

from reacher import reacher_robot_utils

KP = 5.0  # Amps/rad
KD = 2.0  # Amps/(rad/s)
MAX_CURRENT = 3.0  # Amps

serial_port = reacher_robot_utils.get_serial_port()
hardware_interface = interface.Interface(serial_port)
time.sleep(0.25)
hardware_interface.set_joint_space_parameters(kp=KP,
                                              kd=KD,
                                              max_current=MAX_CURRENT)

hardware_interface.read_incoming_data()
hardware_interface.set_actuator_postions(np.zeros([3, 4]))
