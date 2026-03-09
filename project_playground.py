import csdl_alpha as cs
import numpy as np
from scipy.spatial.transform import Rotation as rot

# Design Variables
# Kinematics
kinematics = cs.VariableGroup()
# j1_pos = cs.Variable(name='j1_pos', shape=(3, 1), value=0)
j1_ang = cs.Variable(name='j1_ang', shape=(3, 1), value=0)

j2_pos = cs.Variable(name='j2_pos', shape=(3, 1), value=0)
j2_ang = cs.Variable(name='j2_ang', shape=(3, 1), value=0)

j3_pos = cs.Variable(name='j3_pos', shape=(3, 1), value=0)
j3_ang = cs.Variable(name='j3_ang', shape=(3, 1), value=0)

kinematics.j1_ang = j1_ang
kinematics.j2_pos = j2_pos
kinematics.j2_ang = j2_ang
kinematics.j3_pos = j3_pos
kinematics.j3_ang = j3_ang

# Power
power = cs.VariableGroup()
j1_pow = cs.Variable(name='j1_pow', shape = (1,), value=0)
j2_pow = cs.Variable(name='j2_pow', shape = (1,), value=0)
j3_pow = cs.Variable(name='j3_pow', shape = (1,), value=0)
power.j1_pow = j1_pow
power.j2_pow = j2_pow
power.j3_pow = j3_pow

# Gearing
gearing = cs.VariableGroup()
j1_gear = cs.Variable(name='j1_gear', shape=(1,), value=0)
j2_gear = cs.Variable(name='j2_gear', shape=(1,), value=0)
j3_gear = cs.Variable(name='j3_gear', shape=(1,), value=0)

# Link Geometry
geom = cs.VariableGroup()
l1_d = cs.Variable(name='l1_d', shape=(1,), value=0)
l2_d = cs.Variable(name='l2_d', shape=(1,), value=0)

l1_len = cs.Variable(name='l1_len', shape=(1,), value=0)
l2_len = cs.Variable(name='l2_len', shape=(1,), value=0)

l1_wt = cs.Variable(name='l1_wt', shape=(1,), value=0)
l2_wt = cs.Variable(name='l2_wt', shape=(1,), value=0)

# Constants
al_density = cs.Variable(name='al_density', value=2700)     # [kg/m3]
max_power = cs.Variable(name='max_power', value=500)        # [W]
max_mass = cs.Variable(name='max_mass', value=100)          # [kg]
min_mass = cs.Variable(name='min_mass', value=50)           # [kg]
payload_mass = cs.Variable(name='payload_mass', value=5)    # [kg]



def calc_actuator_mass(power):
    """Returns equivalent actuator mass based on power. Heuristic 

    Args:
        power (_type_): _description_
    """
    pass

def calc_actuator_torque_and_speed(power, gear):
    """Returns max actuator torque and speed based on given power and gear ratio

    Args:
        power (_type_): _description_
    """
    # Efficiency
    nu = 0.6
    mechanical_power = power * nu
    pass

class Robot:
    def __init__(self, kinematics: cs.VariableGroup):
        self.screw_axes
        self.eoat_transform
        pass

    def calc_screw_axis(self, joint_pos, joint_ang):
        # v = cs.cross(-omega, q)
        pass

    def jacobian(self, theta):
        pass

    def inverse_kinematics(self, goal_transform):
        pass

    def forward_kinematics(self, theta):
        pass

def skew_sym(twist: cs.Variable):
    mat = cs.Variable(shape=(4,4), value=0)
    w1 = twist.value[0]
    w2 = twist.value[1]
    w3 = twist.value[2]
    mat = mat.set(cs.slice[0, 1], -w3)
    mat = mat.set(cs.slice[0, 2], w2)
    mat = mat.set(cs.slice[0, 3], twist.value[3])

    mat = mat.set(cs.slice[1, 0], w3)
    mat = mat.set(cs.slice[1, 2], -w1)
    mat = mat.set(cs.slice[1, 3], twist.value[4])

    mat = mat.set(cs.slice[2, 0], -w2)
    mat = mat.set(cs.slice[2, 1], w1)
    mat = mat.set(cs.slice[2, 3], twist.value[5])

    mat = mat.set(cs.slice[-1, -1], 1)

    return mat
    # j3_ang = j3_ang.set(cs.slice[2], cs.sin(theta))


# Load in robot
# Calculate state variables (mass, inertia(?), transforms)
# For each waypoint:
#   Perform IK. Constrain error to be small for each one 
# Construct control spline in joint space 
# Do TOPPRA to get time-optimal trajectory 
# Return 