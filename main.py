import numpy as np
import csdl_alpha as csdl
import utils
from Robot import Robot
import time
import matplotlib.pyplot as plt

########################
### USER INPUTS HERE ###
########################

# XYZ locations of end-effector. Assume zero orientation change
# Ideal solution is a SCARRA config where all joint axes are parallel and along global z-axis
# [3 x n]
all_target_pos = np.array([
    [0.1, 0.1, 0],      # Waypoint 1: robot can reach near origin
    [2.9, 0, 0],        # Waypoint 2: robot can reach full-extents in x
    [0, 2.9, 0],        # Waypoint 3: robot can reach full-extents in y
    [-1.2, 0.5, 0]      # Waypoint 4: robot can reach point inside workspace 
]).T
all_target_ang = np.array([0, np.pi/2, np.pi/3, 0])
use_inline=True

##########################





n_joints = 3
n = all_target_pos.shape[1]

zrot = lambda theta: np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])

target_transforms = np.zeros((4, 4, n))
for i in range(n):
    target_transforms[:3, :3, i] = zrot(all_target_ang[i])
    target_transforms[:3, 3, i] = all_target_pos[:, i]
    target_transforms[3, 3, i] = 1


recorder = csdl.Recorder(inline=use_inline)

j_ang_bounds = np.full((n_joints,), 2*np.pi)
recorder.start()

### Initialize Problem Variables ###
target_transforms = csdl.Variable(value=target_transforms, name='target_transforms')

### Initialize Design Variables ###
j1_pos = csdl.Variable(name='j1_pos', shape=(3,), value=0)
j1_ang = csdl.Variable(name='j1_ang', shape=(3, ), value=np.array([0.1, 0.1, 0.1]))
j1_ang.set_as_design_variable(lower=-j_ang_bounds, upper=j_ang_bounds)

j2_pos = csdl.Variable(name='j2_pos', shape=(3, ), value=np.array([1.5, 0, 0]))
j2_ang = csdl.Variable(name='j2_ang', shape=(3, ), value=np.array([np.pi/2, -np.pi/2, np.pi/4]))
j2_pos.set_as_design_variable()
j2_ang.set_as_design_variable(lower=-j_ang_bounds, upper=j_ang_bounds)

j3_pos = csdl.Variable(name='j3_pos', shape=(3, ), value=np.array([0, 1.5, 0]))
j3_ang = csdl.Variable(name='j3_ang', shape=(3, ), value=np.array([np.pi/2, -np.pi/2, np.pi/4]))
j3_pos.set_as_design_variable()
j3_ang.set_as_design_variable(lower=-j_ang_bounds, upper=j_ang_bounds)

j_ang = csdl.Variable(value=np.array([j1_ang.value, j2_ang.value, j3_ang.value]))
j_pos = csdl.Variable(value=np.array([j1_pos.value, j2_pos.value, j3_pos.value]))

a_power = csdl.Variable(name='actuator_power', value=np.array([10, 10, 10]))
a_gear = csdl.Variable(name='actuator_gear_ratio', value=np.array([1, 1, 1]))
l_id = csdl.Variable(name='link_inner_diameter', value=np.array([0.2, 0.2, 0]))
l_t = csdl.Variable(name='link_thickness', value=np.array([0.01, 0.01, 0]))
a_power.set_as_design_variable(lower=np.full((n_joints,), 0.1), upper=np.full((n_joints,), 100))
a_gear.set_as_design_variable(lower=np.full((n_joints,), 1), upper=np.full((n_joints,), 100))
l_id.set_as_design_variable(lower=np.full((n_joints,), 0.05), upper=np.full((n_joints,), 0.5))
l_t.set_as_design_variable(lower=np.full((n_joints,), 0.001), upper=np.full((n_joints,), 0.1))

robot = Robot(j_ang, j_pos, 
                a_power, a_gear, 
                l_id, l_t)

### Do the things ###
q_ref, kinematic_err = robot.calc_waypoints(target_transforms)
t, qt, qtt, tau = robot.calc_traj(q_ref)

### Set Constraints ###
total_mass = robot.total_mass
total_power = robot.total_power
total_mass.add_name('total_mass')
total_power.add_name('total_power')
total_power.set_as_constraint(upper=200)
total_mass.set_as_constraint(lower=10)

kinematic_err.add_name('kinematic_error')
kinematic_err.set_as_constraint(upper=1e-2)

torque_const = robot.torque_lims - csdl.maximum(utils.abs(tau), axes=(1,), rho=100)
torque_const.add_name('torque_constraint')
torque_const.set_as_constraint(upper=1e-2)

vel_const = robot.vel_lims - csdl.maximum(utils.abs(qt), axes=(1,), rho=100)
vel_const.add_name('velocity_constraint')
vel_const.set_as_constraint(upper=1e-2)

### Set Objective ###
total_time = t[-1]
total_time.add_name('total_time')
total_time.set_as_objective()

recorder.stop()
if use_inline:
    jax_sim = csdl.experimental.JaxSimulator(
        recorder=recorder,
        additional_inputs= [
            j1_ang,
            j2_pos,
            j2_ang,
            j3_pos,
            j3_ang,
            a_power,
            a_gear,
            l_id,
            l_t,
        ],
        additional_outputs=[
            total_time
        ]
    )
    
    t0 = time.time()
    jax_sim.run()
    print('running time:', time.time() - t0)
