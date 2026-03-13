import csdl_alpha as csdl
import numpy as np
import dynamics
import utils
import cubic_spline
from Robot import Robot

def toppra(s, q, qdot, qddot, torque_lim, vel_lim,
           link_frames, screw_axes, inertias, grav):
    n_joints = q.shape[0]
    n = s.shape[0]

    # Coeffs for dynamics equations 
    a = csdl.Variable(shape=(n_joints, n), value=0)
    b = csdl.Variable(shape=(n_joints, n), value=0)
    c = csdl.Variable(shape=(n_joints, n), value=0)

    tau = csdl.Variable(shape=(n_joints, n), value=0)

    zero = csdl.Variable(shape=(n_joints, ), value=0)
    for i in csdl.frange(n):
        ci = dynamics.inverse_dynamics(q, zero, zero, link_frames, screw_axes, inertias, grav)
        c = c.set(csdl.slice[:, i], ci)

        ai = dynamics.inverse_dynamics(q, zero, qdot, link_frames, screw_axes, inertias, grav)
        ai -= ci
        print('a-sub i:', ai.value)
        print('\n')
        a = a.set(csdl.slice[:, i], ai)

        bi = dynamics.inverse_dynamics(q, qdot, qddot, link_frames, screw_axes, inertias, grav)
        bi -= ci
        b = b.set(csdl.slice[:, i], bi)

    return a, b, c

def calc_accel_bounds(x, a, b, c, torque_lim):
    # x = sdot**2
    n_joints = torque_lim.shape[0]

    upper_lim = csdl.Variable(shape=(n_joints, ), value=0)
    lower_lim = csdl.Variable(shape=(n_joints, ), value=0)

    # NOTE: neat trick with smooth abs here
    # Sigmoid logic to deal with sign flipping. Valid because if a[i] 
    # is close to zero (where our sigmoid is not very accurate), acceleration
    # along s requires very little torque, which means that it is unlikely that 
    # that particular joint is driving the overall acceleration limit. 

    for i in csdl.frange(n_joints):
        upper = (torque_lim[i] - b[i] * x - c[i]) / utils.smooth_abs(a[i])
        lower = -(torque_lim[i] - b[i] * x - c[i]) / utils.smooth_abs(a[i])
        print('upper:', upper.value)
        print('lower:', lower.value)
        print('\n')
        upper_lim = upper_lim.set(csdl.slice[i], upper)
        lower_lim = lower_lim.set(csdl.slice[i], lower)
    
    print(upper_lim. value, lower_lim.value)
    amax = csdl.maximum(upper_lim, rho=20)
    amin = csdl.maximum(lower_lim, rho=-20)

    return amax, amin
    



# Convert Robot limits to path constraints

# Compute feasible accel bounds:
# x_new = x_old + 2*ds * sddot
# Comptue bounds on x_new. 

# Backwards pass
# At point s, what speeds let you reach the end goal?

# Forward pass 
# For each s_i, choose the max feasible velocity

# Integration to find time 



if __name__ == "__main__":
    import numpy as np
    import csdl_alpha as csdl
    import cubic_spline
    import utils
    import dynamics
    import modern_robotics as mr

    # 4 waypoints, 3 joints
    q_ref = np.array([
        [0.0, 0.0, 0.25],
        [0.5, 1.5, 0.0],
        [1.2, 0.5, 3.0],
        [1.0, 2.0, 2.0],
        [2.0, 0.0, 2.0],
        [2.0, 0.25, 1.5]
    ])

    s_ref = np.linspace(0, 1, q_ref.shape[0]) # Normalize path 0 to 1

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    j1_pos = csdl.Variable(name='j1_pos', shape=(3,), value=0)
    j1_ang = csdl.Variable(name='j1_ang', shape=(3, ), value=np.array([0.1, 0.1, 0.1]))

    j2_pos = csdl.Variable(name='j2_pos', shape=(3, ), value=np.array([1.5, 0, 0]))
    j2_ang = csdl.Variable(name='j2_ang', shape=(3, ), value=np.array([np.pi/2, -np.pi/2, np.pi/4]))

    j3_pos = csdl.Variable(name='j3_pos', shape=(3, ), value=np.array([0, 1.5, 0]))
    j3_ang = csdl.Variable(name='j3_ang', shape=(3, ), value=np.array([np.pi/2, -np.pi/2, np.pi/4]))

    j_ang = csdl.Variable(value=np.array([j1_ang.value, j2_ang.value, j3_ang.value]))
    j_pos = csdl.Variable(value=np.array([j1_pos.value, j2_pos.value, j3_pos.value]))
    a_power = csdl.Variable(value=np.array([10, 10, 10]))
    a_gear = csdl.Variable(value=np.array([1, 1, 1]))
    l_id = csdl.Variable(value=np.array([0.2, 0.2, 0]))
    l_t = csdl.Variable(value=np.array([0.01, 0.01, 0]))


    robot = Robot(j_ang, j_pos, 
                 a_power, a_gear, 
                 l_id, l_t)
    

    q_ref = csdl.Variable(value=q_ref.T)
    s_ref = csdl.Variable(value=s_ref)
    spline_fit = cubic_spline.fit_cubic_spline(s_ref, q_ref)
    s, q, qdot, qddot  = cubic_spline.discretize_spline(s_ref, q_ref, spline_fit)
    ii = 3
    s_dot = 0.22
    s_ddot = 0.12

    grav = csdl.Variable(value=np.array([0, 0, -9.81]))

  
    a, b, c = toppra(s[ii], q[:, ii], qdot[:, ii], qddot[:, ii], 0, 0, 
                     robot.link_to_link_frames, robot.screw_axes, robot.inertias, grav)
    

    # print(a.value, b.value, c.value)
    

    
    tau1 = dynamics.inverse_dynamics(q[:, ii], qdot[:, ii]*s_dot, qddot[:, ii]*s_dot**2 + qdot[:, ii] * s_ddot, 
                                     robot.link_to_link_frames, robot.screw_axes, robot.inertias, grav)
    tau2 = a * s_ddot + b*s_dot**2 + c

    torque_lim = csdl.Variable(value=np.array([10, 10, 10]))
    # amax, amin = calc_accel_bounds(0.1, a, b, c, torque_lim)
    tau = mr.InverseDynamics(q[:, ii].value, (qdot[:, ii]*s_dot).value, (qddot[:, ii]*s_dot**2 + qdot[:, ii] * s_ddot).value,
                             grav.value, np.zeros((6,)), 
                             robot.link_to_link_frames.value, robot.inertias.value, robot.screw_axes.value.T)
    recorder.stop()

    print('csdl tau:', tau1.value)
    # print('csdl tau2:', tau2.value)
    # print(amax.value, amin.value)
    # tau = mr.InverseDynamics(q[:, ii].value, q[:, ii].value, qdot[:, ii].value, np.array([0, 0, 0]), 0, robot.link_to_link_frames.value, robot.inertias.value, robot.screw_axes.value)

    print('mr tau:', tau)
    