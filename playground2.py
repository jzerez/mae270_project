import numpy as np
import modopt as mo
import csdl_alpha as csdl
import matplotlib.pyplot as plt
import utils
import kinematics
from create_urdf import create_urdf

inline_test = False
recorder = csdl.Recorder(inline=True)

# XYZ locations of end-effector. Assume zero orientation change
# Ideal solution is a SCARRA config where all joint axes are parallel and along global z-axis
# [3 x n]
all_target_pos = np.array([
    [0.1, 0.1, 0],      # Waypoint 1: robot can reach near origin
    [2.9, 0, 0],        # Waypoint 2: robot can reach full-extents in x
    [0, 2.9, 0],        # Waypoint 3: robot can reach full-extents in y
    [-1.2, 0.5, 0]      # Waypoint 4: robot can reach point inside workspace 
]).T


recorder.start()
### Define Design Variables ###
j1_ang = csdl.Variable(name='j1_ang', shape=(3, ), value=np.array([0.1, 0.1, 0.1]))

j2_pos = csdl.Variable(name='j2_pos', shape=(3, ), value=np.array([1.5, 0, 0]))
j2_ang = csdl.Variable(name='j2_ang', shape=(3, ), value=np.array([np.pi/2, -np.pi/2, np.pi/4]))

j3_pos = csdl.Variable(name='j3_pos', shape=(3, ), value=np.array([0, 1.5, 0]))
j3_ang = csdl.Variable(name='j3_ang', shape=(3, ), value=np.array([np.pi/2, -np.pi/2, np.pi/4]))

j1_ang.set_as_design_variable(lower=np.full((3,), -2 * np.pi), upper=np.full((3,), 2 * np.pi))

j2_pos.set_as_design_variable()
j2_ang.set_as_design_variable(lower=np.full((3,), -2 * np.pi), upper=np.full((3,), 2 * np.pi))

j3_pos.set_as_design_variable()
j3_ang.set_as_design_variable(lower=np.full((3,), -2 * np.pi), upper=np.full((3,), 2 * np.pi))

### Initialize Objective ###
total_error = csdl.Variable(name='total_error', shape=(1,), value=0)
total_error.set_as_objective()

### Define Constraints ###
l1 = csdl.norm(j2_pos)
l2 = csdl.norm(j3_pos - j2_pos)
lt = l1 + l2

l1.add_name('length_1')
l2.add_name('length_2')
lt.add_name('total_length')

l1.set_as_constraint(lower=0.01)
l2.set_as_constraint(lower=0.01)
lt.set_as_constraint(upper=3.0)

### Define Intermediate Working Variables ###
n_joints = 3

R_1 = utils.euler_angle_to_rotation_matrix(j1_ang)
frame_1 = csdl.Variable(value=np.identity(4))
frame_1 = frame_1.set(csdl.slice[:3, :3], R_1)
screw_1 = utils.joint_transform_to_screw_axis(frame_1)

R_2 = utils.euler_angle_to_rotation_matrix(j2_ang)
frame_2 = csdl.Variable(value=np.identity(4))
frame_2 = frame_2.set(csdl.slice[:3, :3], R_2)
frame_2 = frame_2.set(csdl.slice[:3, -1], j2_pos)
frame_2_world = csdl.matmat(frame_1, frame_2)
screw_2 = utils.joint_transform_to_screw_axis(frame_2_world)

R_3 = utils.euler_angle_to_rotation_matrix(j3_ang)
frame_3 = csdl.Variable(value=np.identity(4))
frame_3 = frame_3.set(csdl.slice[:3, :3], R_3)
frame_3 = frame_3.set(csdl.slice[:3, -1], j3_pos)
frame_3_world = csdl.matmat(frame_2_world, frame_3)
screw_3 = utils.joint_transform_to_screw_axis(frame_3_world)

screws = csdl.Variable(shape=(6, n_joints), value=0, name='screws')
screws = screws.set(csdl.slice[:, 0], screw_1)
screws = screws.set(csdl.slice[:, 1], screw_2)
screws = screws.set(csdl.slice[:, 2], screw_3)

ee_frame = csdl.copyvar(frame_3_world)
ee_frame.add_name('ee_frame')


### Calculate Error ###
all_target_pos = csdl.Variable(value=all_target_pos)


for i in csdl.frange(all_target_pos.shape[1]):
    ik = kinematics.InverseKinematics(screws, ee_frame, 'ik')
    target_pos = all_target_pos[:, i]
    target_transform = csdl.Variable(value=np.identity(4))
    target_transform = target_transform.set(csdl.slice[:3, -1], target_pos)

    ik_input = csdl.VariableGroup()
    theta = csdl.ImplicitVariable(value=np.zeros((n_joints, )), name='theta' + str(i))
    ik_input.theta = theta
    ik_input.goal = target_transform

    ik_err = ik.evaluate(ik_input).err
    solver = csdl.nonlinear_solvers.Newton(name='newton'+str(i), print_status=False)
    solver.add_state(theta, ik_err, tolerance=1e-4)
    solver.run()

    curr_transform = utils.forward_kinematics_screw(screws, theta, ee_frame)
    curr_error = utils.calc_twist_err(curr_transform, target_transform)
    total_error = total_error + csdl.norm(curr_error)

    

recorder.stop()

if inline_test:
    print(total_error.value)
else:
    from csdl_alpha.experimental import PySimulator, JaxSimulator

    # Create a Simulator object from the Recorder object
    sim = JaxSimulator(
        recorder=recorder,
        additional_inputs=[
            j1_ang,
            j2_pos,
            j2_ang,
            j3_pos,
            j3_ang
        ],
        additional_outputs=[
            total_error
        ]
    )

    # Import CSDLAlphaProblem from modopt
    from modopt import CSDLAlphaProblem

    # Instantiate your problem using the csdl Simulator object and name your problem
    prob = CSDLAlphaProblem(
        problem_name='robot_kinematics',
        simulator=sim,
    )


    from modopt import SLSQP

    # Setup your preferred optimizer (SLSQP) with the Problem object 
    # Pass in the options for your chosen optimizer
    optimizer = SLSQP(prob, solver_options={'maxiter':50})

    # Check first derivatives at the initial guess, if needed
    optimizer.check_first_derivatives(prob.x0)

    # Solve your optimization problem
    optimizer.solve()

    # Print results of optimization
    optimizer.print_results(optimal_variables=True)
    # print(optimizer.results)

    print('j1_ang:', j1_ang.value)
    print('j2_pos:', j2_pos.value)
    print('j2_ang:', j2_ang.value)
    print('j3_pos:', j3_pos.value)
    print('j3_ang:', j3_ang.value)


    my_configs = [
        {'xyz': [0, 0, 0], 'rpy': j1_ang.value},        # Joint 0
        {'xyz': j2_pos.value, 'rpy': j2_ang.value},  # Joint 1
        {'xyz': j3_pos.value, 'rpy': j3_ang.value},        # Joint 2
    ]

    robot = create_urdf(my_configs, 'p2.urdf')