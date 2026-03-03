import numpy as np
import csdl_alpha as csdl

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
    r_total = csdl.matmat(rz, csdl.matmat(ry, rx))
    
    return r_total

def rotation_matrix_to_euler_angle(rotation):
    # 1. Extract the Pitch (theta)
    # R[2, 0] corresponds to -sin(theta) in the ZYX matrix
    # We clip the value to avoid NaNs from floating point noise outside [-1, 1]
    r20 = rotation[2, 0]
    pitch = -csdl.arcsin(r20)

    # 2. Extract Roll (phi) and Yaw (psi) using atan2
    # Standard ZYX mapping:
    # R[2, 1] = sin(phi)cos(theta)
    # R[2, 2] = cos(phi)cos(theta)
    # R[1, 0] = sin(psi)cos(theta)
    # R[0, 0] = cos(psi)cos(theta)
    
    roll = csdl.arctan(rotation[2, 1] / rotation[2, 2])
    yaw = csdl.arctan(rotation[1, 0] / rotation[0, 0])

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
    t_inv = csdl.Variable(shape=(4, 4), value=np.identity(4))

    r_inv = csdl.transpose(transform[:3, :3])
    p_inv = csdl.matmat(-r_inv, transform[:3, -1])

    t_inv = t_inv.set(csdl.slice[:3, :3],  r_inv)
    t_inv = t_inv.set(csdl.slice[:3, -1],  p_inv)

    return t_inv

def rotation_exp(omega, theta):
    omega_ss = vec_to_skew_symmetric(omega)
    rotation = csdl.Variable(value=np.identity(3))
    rotation += csdl.sin(theta) * omega_ss
    rotation += (1 - csdl.cos(theta)) * csdl.matmat(omega_ss, omega_ss)

    return rotation

def transform_exp(screw, theta):
    omega = screw[:3]
    v = screw[3:]

    omega_ss = vec_to_skew_symmetric(omega)

    r = rotation_exp(omega, theta)
    k = csdl.Variable(value=np.identity(3)) * theta
    k += (1 - csdl.cos(theta)) * omega_ss
    k += (theta - csdl.sin(theta)) * csdl.matmat(omega_ss, omega_ss)

    p = csdl.matvec(k, v)

    transform = csdl.Variable(value=np.identity(4))

    transform = transform.set(csdl.slice[:3, :3], r)
    transform = transform.set(csdl.slice[:3, -1], p)

    return transform

def rotation_log(rotation):
    if trace(rotation).value == 3:
        theta = csdl.Variable(shape=(1,), value=0)
        omega = csdl.Variable(shape=(3,), value=0)
        return omega, theta
    # elif trace(rotation) == -1:
    #     theta = csdl.Variable(shape=(1,), value=0)
    #     omega = 
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
    # Check to make sure omega is unit length
    assert(np.isclose(csdl.norm(omega).value, 1))

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

if __name__ == "__main__":
    import numpy as np
    import csdl_alpha as csdl
    import modern_robotics as mr
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    T = np.array([[ 1,    -0,     0,     0.26 ],
                    [-0,     1,     0,     0.043],
                    [ 0,     0,     1,     0   ],
                    [ 0,     0,     0,     1   ]])
    
    omega, theta = transform_log(csdl.Variable(value=T))
    twist = theta*omega
    recorder.stop()

    print(omega.value, theta.value, twist.value)                    
    print(mr.se3ToVec(mr.MatrixLog6(T)))
