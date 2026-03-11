import numpy as np
import csdl_alpha as csdl
import modern_robotics as mr
import utils

# Inverse Dynamics: Given q, qdot, qddot; find torque
def inverse_dynamics(q, qdot, qddot, link_frames, screw_axes, inertias, grav=csdl.Variable(value=np.array([0, 0, -9.81]))):
    """Calculate the torque required for a given joint-space pos/vel/acc
    
    Args:
        q (csdl.Variable): (n,) Vector of joint positions [rad]
        qdot (csdl.Variable): (n,) Vector of joint velocities [rad/s]
        qddot (csdl.Variable): (n,) Vector of joint accelerations [rad/s/s]
        link_frames (csdl.Variable): (n, 4, 4) Tensor of link frames. Each frame
            is defined at the link's center of mass and is defined relative to 
            the frame of the downstream link (away from the robot base)
        screw_axes (csdl.Variable): (6, n) Screw axes for each joint when 
            the robot is in the home position. relative to the world frame
        inertias (csdl.Variable): (n, 6, 6) Tensor of link inertias
        grav (csdl.Variable, optional): (3,) Vector representing gravity
            Defaults to csdl.Variable(value=np.array([0, 0, -9.81])).
    """
    n_joints = 3
    n_links = 3

    link_configs = csdl.Variable(shape=(n_links, 4, 4), value=0)
    link_vels = csdl.Variable(shape=(n_links, 6), value=0)
    link_accels = csdl.Variable(shape=(n_links, 6), value=0)
    torques = csdl.Variable(shape=(n_joints,), value=0)

    # Transform Screw Axes to link frames. (n, 6) to match other vars
    axes = csdl.Variable(shape=csdl.transpose(screw_axes).shape, value=0)

    for i in range(n_links):
        axes[i, :] = csdl.matmat(utils.adjoint(link_frames[i, :, :]), screw_axes[:, i])
    
    
    Vi_prev = csdl.Variable(shape=(6,), value=0)
    Vi_dot_prev = csdl.Variable(shape=(6,), value=0)
    F_upstream = csdl.Variable(shape=(6,), value=0)

    for i in range(n_links):
        # Transform of link i, expressed in link i-1's frame
        Ti = csdl.matmat(utils.transform_exp(axes[i, :], -q[i]), link_frames[i, :, :])

        # Twist/Velocity of link i, expressed in link i's frame
        # First term is the velocity due to the upstream joint's motion
        # Second term is the velocity due to joint i's rotation
        Vi = csdl.matmat(utils.adjoint(Ti), Vi_prev) + axes[i, :] * qdot[i]
        Vi_prev = Vi

        # Spatial Accel of link i
        # First term is accel due to upstream joint
        # Second term is accel due to joint i's rotation
        # Third term is accel due to corriolis forces 
        Vi_dot  = csdl.matmat(utils.adjoint(Ti), Vi_dot_prev) \
            + axes[i, :] * qddot[i] \
            + qdot[i] * csdl.matmat(utils.lie_bracket(Vi), axes[i, :])
        Vi_dot_prev = Vi_dot
        
        link_configs[i] = Ti
        link_vels[i] = Vi
        link_accels[i] = Vi_dot

    for i in range(n_joints-1, -1, -1):
        Ti = utils.invert_transform(link_configs[i, :])
        Vi = link_vels[i, :]
        Vi_dot = link_accels[i, :]
        Gi = inertias[i, :, :]

        Fi = csdl.matmat(csdl.transpose(utils.adjoint(Ti)), F_upstream)\
            + csdl.matmat(Gi, Vi_dot)\
            - csdl.matmat(csdl.matmat(csdl.transpose(utils.lie_bracket(Vi)), Gi), Vi)
        
        torques[i] = csdl.matmat(csdl.transpose(Fi), axes[i, :])
    
    return torques


if __name__ == "__main__":
    # test code here
    pass