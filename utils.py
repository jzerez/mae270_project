import numpy as np
import csdl_alpha as csdl

def unit(vec: csdl.Variable):
    return vec / csdl.norm(vec)

def linspace(x0: csdl.Variable, x1:csdl.Variable, n: int, endpoint: bool = True):
    # If-statement is OK if the value doesn't change after compile time
    res = csdl.Variable(shape=(n,), value=0, name='linspace')
    step = (x1 - x0)/(n-endpoint)
    step.add_name('step_size')

    for i in csdl.frange(n):
        res = res.set(csdl.slice[i], x0 + step*i)
    return res

def smooth_abs(x, k=20):
    operation = 1 / (1 + csdl.exp(-k*x)) * 2 - 1
    return x * operation


def abs(x, eps=1e-10):
    return csdl.sqrt(x**2 + eps)

def safe_max(x, axes=None, rho=20):
    n = x.size
    if not axes is None:
        for i in axes:
            n /= x.shape[i]
    return csdl.maximum(x, axes=axes, rho=rho) - csdl.log(n/rho)

def safe_min(x, axes=None, rho=20):
    n = x.size
    if not axes is None:
        for i in axes:
            n /= x.shape[i]
    return csdl.maximum(x, axes=axes, rho=rho) + csdl.log(n/rho)


def euler_angle_to_rotation_matrix(angle: csdl.Variable):
    """Convert euler angles (roll, pitch, yaw) into a rotation matrix
    Applies using the ZYX rotation order 

    Args:
        angle (csdl.Variable): vector of roll, pitch, yaw. [radians]

    Returns:
        mat (csdl.Variable): a (3,3) rotation matrix
    """

    rx = rotate_x(angle[0])
    ry = rotate_y(angle[1])
    rz = rotate_z(angle[2])

    # Compute total rotation
    mat = csdl.matmat(rz, csdl.matmat(ry, rx))
    
    return mat

def rotation_matrix_to_euler_angle(mat: csdl.Variable):
    """Compute euler angles from a 3x3 rotation matrix 
    Applies using the ZYX rotation order

    Args:
        mat (csdl.Variable): vector of roll, pitch, yaw [rad]

    Returns:
        _type_: _description_
    """
    # 1. Extract the Pitch (theta)
    # R[2, 0] corresponds to -sin(theta) in the ZYX matrix
    pitch = -csdl.arcsin(mat[2, 0])

    # 2. Extract Roll (phi) and Yaw (psi) using atan2
    # Standard ZYX mapping:
    # R[2, 1] = sin(phi)cos(theta)
    # R[2, 2] = cos(phi)cos(theta)
    # R[1, 0] = sin(psi)cos(theta)
    # R[0, 0] = cos(psi)cos(theta)
    
    roll = csdl.arctan(mat[2, 1] / mat[2, 2])
    yaw = csdl.arctan(mat[1, 0] / mat[0, 0])

    angle = csdl.Variable(shape=(3,), value=0)
    angle = angle.set(csdl.slice[0], roll)
    angle = angle.set(csdl.slice[1], pitch)
    angle = angle.set(csdl.slice[2], yaw)

    return angle

def rotate_x(angle: csdl.Variable):
    """Returns rotation matrix for a rotation about x axis

    Args:
        angle (csdl.Variable): Rotation angle [radians]

    Returns:
        rx (csdl.Variable): 3x3 rotation matrix
    """
    rx = csdl.Variable(value=np.identity(3))
    rx = rx.set(csdl.slice[1, 1],  csdl.cos(angle))
    rx = rx.set(csdl.slice[1, 2], -csdl.sin(angle)) 
    rx = rx.set(csdl.slice[2, 1],  csdl.sin(angle))
    rx = rx.set(csdl.slice[2, 2],  csdl.cos(angle))

    return rx

def rotate_y(angle: csdl.Variable):
    """Returns rotation matrix for a rotation about y axis

    Args:
        angle (csdl.Variable): Rotation angle [radians]

    Returns:
        ry (csdl.Variable): 3x3 rotation matrix
    """
    ry = csdl.Variable(value=np.identity(3))
    ry = ry.set(csdl.slice[0, 0],  csdl.cos(angle))
    ry = ry.set(csdl.slice[0, 2],  csdl.sin(angle))
    ry = ry.set(csdl.slice[2, 0], -csdl.sin(angle)) 
    ry = ry.set(csdl.slice[2, 2],  csdl.cos(angle))

    return ry

def rotate_z(angle: csdl.Variable):
    """Returns rotation matrix for a rotation about z axis

    Args:
        angle (csdl.Variable): Rotation angle [radians]

    Returns:
        rz (csdl.Variable): 3x3 rotation matrix
    """
    rz = csdl.Variable(value=np.identity(3))
    rz = rz.set(csdl.slice[0, 0],  csdl.cos(angle))
    rz = rz.set(csdl.slice[1, 0],  csdl.sin(angle))
    rz = rz.set(csdl.slice[0, 1], -csdl.sin(angle)) 
    rz = rz.set(csdl.slice[1, 1],  csdl.cos(angle))

    return rz

def invert_transform(transform: csdl.Variable):
    """Calculates the inverse of a 4x4 Homogenous transform

    Given a transform with rotation R and translation p:
    T = [R, p;
         0, 1]

    Its inverse is given by:
    T^-1 = [R^T, -R^T * p;
              0,         1]

    Args:
        transform (csdl.Variable): (4, 4) transform

    Returns:
        t_inv (csdl.Variable) (4, 4) inverse transform
    """
    t_inv = csdl.Variable(shape=(4, 4), value=np.identity(4))

    r_inv = csdl.transpose(transform[:3, :3])
    p_inv = csdl.matmat(-r_inv, transform[:3, -1])

    t_inv = t_inv.set(csdl.slice[:3, :3],  r_inv)
    t_inv = t_inv.set(csdl.slice[:3, -1],  p_inv)

    return t_inv

def rotation_exp(omega: csdl.Variable, theta: csdl.Variable):
    """Computes the matrix exponential for rotation
    Given an angular displacement (theta) about an axis (omega), calculate
    the equivalent rotation matrix

    Uses Rodriguez's Formula 

    Args:
        omega (csdl.Variable): (3,) vector representing rotation axis
        theta (csdl.Variable): (1,) angular displacement [rad]

    Returns:
        rotation (csdl.Variable): (3, 3) equivalent rotation matrix
    """
    omega_ss = vec_to_skew_symmetric(omega)
    rotation = csdl.Variable(value=np.identity(3))
    rotation += csdl.sin(theta) * omega_ss
    rotation += (1 - csdl.cos(theta)) * csdl.matmat(omega_ss, omega_ss)

    return rotation

def transform_exp(screw: csdl.Variable, theta: csdl.Variable):
    """Computes the matrix exponential for homogeneous transforms

    Given an angular displacement (theta) about a screw axis (screw),
    Calculate the equivalent homogenous transformation matrix

    Args:
        screw (csdl.Variable): (6,) vector representing screw axis. 
        theta (csdl.Variable): (1,) angular displacement [rad]

    Returns:
        transform (csdl.Variable): (4, 4) equivalent homogenous transformation matrix 
    """
    # Extract the angular and linear components of the screw axis
    omega = screw[:3]
    v = screw[3:]

    # Skew symetric form of rotation axis
    omega_ss = vec_to_skew_symmetric(omega)

    # Calculate equivalent rotation matrix
    r = rotation_exp(omega, theta)

    # k is a temp variable to calculate equivalent displacement vector p
    # Modified Rodriguez's formula
    k = csdl.Variable(value=np.identity(3)) * theta
    k += (1 - csdl.cos(theta)) * omega_ss
    k += (theta - csdl.sin(theta)) * csdl.matmat(omega_ss, omega_ss)

    # Equivalent displacement vector p
    p = csdl.matvec(k, v)

    # Load results into transform variable
    transform = csdl.Variable(value=np.identity(4))

    transform = transform.set(csdl.slice[:3, :3], r)
    transform = transform.set(csdl.slice[:3, -1], p)

    return transform

def rotation_log(rotation: csdl.Variable):
    """Computes the matrix logarithm for a rotation matrix

    Given a rotation matrix, calculate an axis and angular displacement 
    that results in equivalent rotation. 

    NOTE: This function uses if-statements, which do not allow for 
    gradient propogation. 

    Args:
        rotation (csdl.Variable): (3, 3) Rotation matrix

    Returns:
        omega (csdl.Variable): (3,) unit axis of rotation
        theta (csdl.Variable): (1,) angular displacement [rad] 
    """
    # Check edge case, where R is the identity matrix
    # If so, theta is zero and omega vector is also zero
    if trace(rotation).value == 3:
        theta = csdl.Variable(shape=(1,), value=0)
        omega = csdl.Variable(shape=(3,), value=0)
        return omega, theta

    # General case
    # NOTE! 
    # Currently, this implementation will fail if theta is exactly 
    # equal to an integer multiple of pi. In future will need more logic to detect
    theta = csdl.arccos(0.5*(trace(rotation)-1))
    omega_ss = 1 / (2 * csdl.sin(theta)) * (rotation - csdl.transpose(rotation))
    omega = skew_symmetric_to_vec(omega_ss)

    return omega, theta

def transform_log(transform):
    screw = csdl.Variable(shape=(6,), value=0)

    rotation = transform[:3, :3]
    p = transform[:3, -1]
    omega, theta = rotation_log(rotation)
    if theta.value == 0:
        theta = csdl.norm(p)
        screw = screw.set(csdl.slice[3:], p / theta)
        return screw, theta
    omega_mat = vec_to_skew_symmetric(omega)

    theta_mat = csdl.Variable(value=np.identity(3)) / theta
    v = theta_mat - 0.5 * omega_mat + (1/theta - 0.5 * (csdl.cos(theta / 2) / csdl.sin(theta / 2))) * csdl.matmat(omega_mat, omega_mat)
    v = csdl.matvec(v, p)

    
    screw = screw.set(csdl.slice[:3], omega)
    screw = screw.set(csdl.slice[3:], v)

    return screw, theta

def trace(mat: csdl.Variable):
    shape = mat.shape
    n = shape[0]
    if shape[0] != shape[1]:
        raise ValueError('Expected square matrix')
    
    res = csdl.Variable(value=0)
    for i in csdl.frange(n):
        res += mat[i, i]
    return res

def vec_to_skew_symmetric(vec):
    mat = csdl.Variable(shape=(3,3), value=0)
    mat = mat.set(csdl.slice[0, 1], -vec[2])
    mat = mat.set(csdl.slice[0, 2], vec[1])
    mat = mat.set(csdl.slice[1, 0], vec[2])
    mat = mat.set(csdl.slice[1, 2], -vec[0])
    mat = mat.set(csdl.slice[2, 0], -vec[1])
    mat = mat.set(csdl.slice[2, 1], vec[0])

    return mat

def skew_symmetric_to_vec(mat):
    vec = csdl.Variable(shape=(3,), value=0)
    vec = vec.set(csdl.slice[0], -mat[1, -1])
    vec = vec.set(csdl.slice[1], mat[0, -1])
    vec = vec.set(csdl.slice[2], -mat[0, 1])
    return vec

def adjoint(transform: csdl.Variable):
    """Computes the adjoint of a 4x4 transformation matrix:
    If T = [R, p
            0, 1]

    Where R is a (3,3) rotation matrix and p is a (3,1) displacement vector

    Ad(T) = [R, 0
             [p]R, R]

    Args:
        transform (csdl.Variable): (4, 4) transform

    Returns:
        mat (csdl.Variable): (6, 6) adjoint matrix. 
    """
    r = transform[:3, :3]
    p = transform[:3, -1]

    mat = csdl.Variable(shape=(6,6), value=0)
    mat = mat.set(csdl.slice[:3, :3], r)
    mat = mat.set(csdl.slice[3:, 3:], r)
    mat = mat.set(csdl.slice[3:, :3], csdl.matmat(vec_to_skew_symmetric(p), r))

    return mat

def joint_transform_to_screw_axis(transform: csdl.Variable):
    """Calculate a joint's screw axis from its transform.
    Assumes Z-axis is axis of rotation

    Args:
        transform (csdl.Variable): (4, 4) transform of joint frame in the world frame

    Returns:
        screw (csdl.Variable): (6, 1) vector representing screw axis. 
    """
    screw = csdl.Variable(value=np.zeros((6,)))

    # Extract rotation axis (z-axis by convention)
    omega = transform[:3, 2]
    # # Check to make sure omega is unit length
    # assert(np.isclose(csdl.norm(omega).value, 1))

    q = transform[:3, -1]

    v = csdl.cross(-omega, q)

    screw = screw.set(csdl.slice[:3], omega)
    screw = screw.set(csdl.slice[3:], v)

    return screw

def calc_jacobian(screws: csdl.Variable, theta: csdl.Variable):
    """Calculate the Jacobian in the world frame

    Args:
        screws (csdl.Variable): (6, n) matrix of screw axes for each actuator in the world frame
        theta (csdl.Variable): (n, ) vector of joint angles for each actuator [rad]

    Returns:
        jac (csdl.Variable): (6, n) matrix
    """
    n = screws.shape[-1]
    jac = csdl.Variable(shape=screws.shape, value=0)
    jac = jac.set(csdl.slice[:, 0], screws[:, 0])

    t = csdl.Variable(value=np.identity(4))
    i = csdl.Variable(value=0, shape=(1,), name='i')
    for i in range(1, n):
        t = csdl.matmat(t, transform_exp(screws[:, i-1], theta[i-1]))
        jac = jac.set(csdl.slice[:, i], csdl.matvec(adjoint(t), screws[:, i]))
    
    return jac

def left_quat_multiply(quat):
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    q = np.array([
        (w, -x, -y, -z), 
        (x,  w, -z,  y),
        (y,  z,  w, -x),
        (z, -y,  x,  w)
    ])

    return csdl.Variable(value=q)

def rotation_matrix_to_quat(mat):
    w = csdl.sqrt(1 + mat[0, 0] + mat[1, 1] + mat[2, 2]) / 2
    x = (mat[2, 1] - mat[1, 2]) / (4 * w)
    y = (mat[0, 2] - mat[2, 0]) / (4 * w)
    z = (mat[1, 0] - mat[0, 1]) / (4 * w)

    quat = csdl.Variable(shape=(4,), value=0)
    quat = quat.set(csdl.slice[0], w)
    quat = quat.set(csdl.slice[1], x)
    quat = quat.set(csdl.slice[2], y)
    quat = quat.set(csdl.slice[3], z)

    return quat

def forward_kinematics(frames: csdl.Variable, thetas: csdl.Variable):
    """Computes robot's end-effector pose given joint angles.
    Based on a robot's set of transforms for each joint

    Args:
        frames (csdl.Variable): (4, 4, n) set of transforms expressing each joint's pose relative
            to its parent joint. Z-axis is rotational axis 
        thetas (csdl.Variable): (n, ) vector of joitn angles [rad]

    Returns:
        total_transform (csdl.Variable): (4, 4) transform for the end-effector in the fixed/world frame
    """
    n = thetas.size
    total_transform = csdl.Variable(value=np.identity(4), name='total_transform')
    ee_frame = csdl.Variable(value=np.identity(4))

    # Update transform for each joint
    for i in csdl.frange(1, n):
        theta = thetas[i-1]
        frame = frames.get(csdl.slice[:, :, i])
        joint_rotation = rotate_z(theta)
        joint_transform = csdl.Variable(value=np.identity(4))
        joint_transform = joint_transform.set(csdl.slice[:3, :3], joint_rotation)
        total_transform = csdl.matmat(total_transform, csdl.matmat(joint_transform, frame))
    
    # Update transform for the last joint 
    joint_rotation = rotate_z(thetas[-1])
    joint_transform = csdl.Variable(value=np.identity(4))
    joint_transform = joint_transform.set(csdl.slice[:3, :3], joint_rotation)
    total_transform = csdl.matmat(total_transform, csdl.matmat(joint_rotation, ee_frame))
    
    return total_transform

def forward_kinematics_screw(screws: csdl.Variable, thetas: csdl.Variable, ee_frame: csdl.Variable):
    """Computes robot's end-effector pose given joint angles. 
    Based on robot's screw axes when it is in the zero pose

    Args:
        screws (csdl.Variable): (6, n) screw axes
        thetas (csdl.Variable): (n,) joint angles [rad]
        ee_frame (csdl.Variable): (4, 4) transform for the end-effector relative to the world-frame
            when the robot is in the zero pose

    Returns:
        total_transform (csdl.Variable): (4, 4) transformation matrix of end-effector pose 
            in the fixed/world frame given joint angles 
    """
    n = thetas.size
    total_transform = csdl.Variable(value=np.identity(4), name='total_transform')

    for i in csdl.frange(n):
        theta = thetas[i]
        screw = screws.get(csdl.slice[:, i])
        total_transform = csdl.matmat(total_transform, transform_exp(screw, theta))

    total_transform = csdl.matmat(total_transform, ee_frame)
    return total_transform

def calc_twist_err(transform: csdl.Variable, goal_transform: csdl.Variable):
    """Calculates the error between the current end effector pose and the goal.
    Returns a 6D twist vector. 

    Args:
        transform (csdl.Variable): (4, 4) Transform for current end-effector pose
        goal_transform (csdl.Variable): (4, 4) goal transform for end-effector

    Returns:
        err (csdl.Variable): (6,) Twist vector expressed in base/static frame
    """
    # Goal transform expressed in end-effector frame
    Tbd = csdl.matmat(invert_transform(transform), goal_transform)

    # Compute matrix log to obtain screw aixs
    screw_axis, theta = transform_log(Tbd)
    
    # scale the unit screw axis to get twist (in end-effector frame)
    twist = screw_axis * theta

    # Project twist back into the base/static frame
    return csdl.matvec(adjoint(transform), twist)

def lie_bracket(twist: csdl.Variable):
    """Calculate the 6x6 matrix representing the Lie-Bracket of the given 6-vector
        The Lie Bracket is a generalization of the cross-product between two 6-vectors

        For a given twist, V = [w, v]

        The lie-bracket will return:
        [[w], 0,
         [v], [w]]

        Where [w] and [v] are the skew symmetric forms of omega and v. 

    Args:
        twist (csdl.Variable): (6,) Spatial twist vector

    Returns:
        ad (csdl.Variable): (6,6) Lie Bracket of the vector
    """
    w_ss = vec_to_skew_symmetric(twist[:3])
    v_ss = vec_to_skew_symmetric(twist[3:])
    ad = csdl.Variable(shape=(6,6), value=0)

    ad = ad.set(csdl.slice[:3, :3], w_ss)
    ad = ad.set(csdl.slice[3:, :3], v_ss)
    ad = ad.set(csdl.slice[3:, 3:], w_ss)
    
    return ad

if __name__ == "__main__":
    import numpy as np
    import csdl_alpha as csdl
    import modern_robotics as mr
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    a = csdl.Variable(shape=(1,), value=1)
    b = csdl.Variable(shape=(1,), value=10)
    n = int(10)

  
    ls1 = linspace(a, b, n, True)
    ls2 = linspace(a, b, n, False)

    recorder.stop()
    recorder.visualize_graph('linspace')
    print(ls1.value)
    print(ls2.value)



    

