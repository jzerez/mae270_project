import numpy as np
import csdl_alpha as csdl

def fit_cubic_spline(s: csdl.Variable, y: csdl.Variable):
    """Computes constants for a piecewise cubic polynomial through the provided waypoints
    Given n_waypoints, the number of piecewise polynomials (n_segments) will be one fewer. 
    The polynomials are constructed to preserve C1 continuity between segments. 

    Function created based on code from Google Gemini. 

    Args:
        s (csdl.Variable): (n_waypoints,) Vector of parametric coordinates for each waypoint
        y (csdl.Variable): (n_joints, n_waypoints) Matrix of desired waypoints in joint coordinates [rad]

    Returns:
        coeffs (csdl.Variable): (n_joints, n_segments, 4) Matrix of polynomial coefficients.
            n_segments is equal to n_waypoints minus 1.  
    """
    n_seg = s.size - 1
    h = s[1:] - s[:-1]

    n_joints = y.shape[0]
    
    # We solve the system Ak = B for each joint
    # A is a tridiagonal matrix shared by all joints
    A = csdl.Variable(shape=(n_seg + 1, n_seg + 1), value=0)

    k = csdl.Variable(shape=(n_seg + 1, n_joints), value=0)
    B = csdl.Variable(shape=(n_seg + 1, n_joints), value=0)
    
    # Natural Spline Boundary Conditions: k0 = 0, kn = 0
    A = A.set(csdl.slice[0, 0], 1)
    A = A.set(csdl.slice[n_seg, n_seg], 1)
    
    # Fill the tridiagonal matrix and the B vector
    # for i in csdl.frange(1, n_seg):
    for i in csdl.frange(1, n_seg):

        A = A.set(csdl.slice[i, i-1], h[i-1])
        A = A.set(csdl.slice[i, i], 2 * (h[i-1] + h[i]))
        A = A.set(csdl.slice[i, i+1], h[i])
        
        # Calculate B for all joints simultaneously
        b = 6 * ((y[:, i+1] - y[:, i]) / h[i] - (y[:, i] - y[:, i-1]) / h[i-1])
        B = B.set(csdl.slice[i], b)
    
    # Solve for the constants
    for i in csdl.frange(n_joints):
        k = k.set(csdl.slice[:, i], csdl.solve_linear(A, B[:, i]))
    

    # Store coefficients for each segment. [n_joint x n_seg x 4]
    # a(s-si)^3 + b(s-si)^2 + c(s-si) + d
    coeffs = csdl.Variable(shape=(n_joints, n_seg, 4), value=0)

    for i in csdl.frange(n_seg):
        a = (k[i+1] - k[i]) / (6 * h[i])
        b = k[i] / 2
        c = (y[:, i+1] - y[:, i]) / h[i] - (h[i] * (2*k[i] + k[i+1]) / 6)
        d = y[:, i]

        coeffs = coeffs.set(csdl.slice[:, i, 0], a)
        coeffs = coeffs.set(csdl.slice[:, i, 1], b)
        coeffs = coeffs.set(csdl.slice[:, i, 2], c)
        coeffs = coeffs.set(csdl.slice[:, i, 3], d)

    return coeffs


def discretize_spline(s, y, coeffs, points_per_segment=25):
    """Create a dense set of discrete waypoints by sampling cubic polynomial

    Args:
        s (csdl.Variable): (n_waypoints,) Vector of parametric coordinates for each waypoint
        y (csdl.Variable): (n_joints, n_waypoints) Matrix of desired waypoints in joint coordinates [rad]
        coeffs (csdl.Variable): (n_joints, n_segments, 4) Matrix of polynomial coefficients.
            n_segments is equal to n_waypoints minus 1.  
        points_per_segment (int, optional): Number of points to generate per segment. Defaults to 25.

    Returns:
        s_fine (csdl.Variable): (n,) Vector of all parametric values used
        waypoints (csdl.Variable): (n_joints, n) Vector of all generated waypoints
    """
    
    n_joints, n_segments, _ = coeffs.shape

    waypoints = csdl.Variable(shape=(n_joints, points_per_segment * n_segments + 1), value=0)
    s_fine = csdl.Variable(shape=(points_per_segment * n_segments + 1,), value=0)

    for i in range(1, n_segments+1):
        # TODO: fix the line below--will not be differentiable
        ds = np.linspace(s[i-1].value, s[i].value, points_per_segment, endpoint=False)
        ds = csdl.Variable(value=ds.flatten())

        start_idx = (i - 1) * points_per_segment
        end_idx = i * points_per_segment

        s_fine = s_fine.set(csdl.slice[start_idx:end_idx], ds)
        ds = ds - s[i-1]

        for j in range(n_joints):
            c = coeffs[j, i-1, :]
            interp_values = c[0]*ds**3 + c[1]*ds**2 + c[2]*ds + c[3]
            waypoints = waypoints.set(csdl.slice[j, start_idx:end_idx], interp_values)

    waypoints = waypoints.set(csdl.slice[:, -1], y[:, -1])
    s_fine = s_fine.set(csdl.slice[-1], s[-1])
    return s_fine, waypoints


def evaluate_spline(s_query, s_waypoints, coeffs):
    """Evaluates the spline at a specific s value."""
    # Find which segment s_query falls into
    if s_query <= s_waypoints[0]: return coeffs[0][3]
    if s_query >= s_waypoints[-1]: 
        # Evaluate last point of last segment
        idx = len(coeffs) - 1
        h = s_waypoints[-1] - s_waypoints[-2]
        a, b, c, d = coeffs[idx]
        return a*h**3 + b*h**2 + c*h + d
    
    idx = np.searchsorted(s_waypoints, s_query) - 1
    ds = s_query - s_waypoints[idx]
    a, b, c, d = coeffs[idx]
    return a*ds**3 + b*ds**2 + c*ds + d



def fit_cubic_spline_2(s, y):
    """
    Fits a natural cubic spline to waypoints.
    s: 1D array of path parameters (e.g., [0, 0.3, 0.7, 1.0])
    y: 2D array of joint angles (N_waypoints x N_joints)
    """
    n = len(s) - 1
    h = np.diff(s)
    num_joints = y.shape[1]
    
    # We solve the system Ak = B for each joint
    # A is a tridiagonal matrix shared by all joints
    A = np.zeros((n + 1, n + 1))
    B = np.zeros((n + 1, num_joints))
    
    # Natural Spline Boundary Conditions: k0 = 0, kn = 0
    A[0, 0] = 1
    A[n, n] = 1
    
    # Fill the tridiagonal matrix and the B vector
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        
        # Calculate B for all joints simultaneously
        B[i] = 6 * ((y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i-1])
        
    # Solve for k (second derivatives)
    k = np.linalg.solve(A, B)
    
    # Store coefficients for each segment: a(s-si)^3 + b(s-si)^2 + c(s-si) + d
    coeffs = []
    for i in range(n):
        a = (k[i+1] - k[i]) / (6 * h[i])
        b = k[i] / 2
        c = (y[i+1] - y[i]) / h[i] - (h[i] * (2*k[i] + k[i+1]) / 6)
        d = y[i]
        coeffs.append((a, b, c, d))
        
    return coeffs


# --- Example Usage ---
# 4 waypoints, 3 joints
way_pts = np.array([
    [0.0, 0.0, 0.25],
    [0.5, 1.5, 0.0],
    [1.2, 0.5, 3.0],
    [1.0, 2.0, 2.0],
    [2.0, 0.0, 2.0],
    [2.0, 0.25, 1.5]
])
s_pts = np.linspace(0, 1, way_pts.shape[0]) # Normalize path 0 to 1
coeffs = fit_cubic_spline_2(s_pts, way_pts) 

recorder = csdl.Recorder(inline=True)
recorder.start()
way_pts = csdl.Variable(value=way_pts.T)
s_pts = csdl.Variable(value=s_pts)
spline_fit = fit_cubic_spline(s_pts, way_pts)
s_fine, path_points = discretize_spline(s_pts, way_pts, spline_fit)

recorder.stop()

import matplotlib.pyplot as plt
plt.figure()
plt.plot(s_pts.value, way_pts.value[0, :], 'C0o', label='J0')
plt.plot(s_pts.value, way_pts.value[1, :], 'C1o', label='J1')
plt.plot(s_pts.value, way_pts.value[2, :], 'C2o', label='J2')

plt.plot(s_fine.value, path_points.value[0, :], 'C0--', label='J0')
plt.plot(s_fine.value, path_points.value[1, :], 'C1--', label='J1')
plt.plot(s_fine.value, path_points.value[2, :], 'C2--', label='J2')
plt.title('CSDL Path Generation')
plt.legend()

plt.figure()
s_fine = np.linspace(0, 1, 100)
path_points2 = np.array([evaluate_spline(s, s_pts.value, coeffs) for s in s_fine])
plt.plot(s_pts.value, way_pts.value[0, :], 'C0o', label='J0')
plt.plot(s_pts.value, way_pts.value[1, :], 'C1o', label='J1')
plt.plot(s_pts.value, way_pts.value[2, :], 'C2o', label='J2')

plt.plot(s_fine, path_points2.T[0, :], 'C0--', label='J0')

plt.plot(s_fine, path_points2.T[1, :], 'C1--', label='J1')

plt.plot(s_fine, path_points2.T[2, :], 'C2--', label='J2')
plt.title('Default Path Generation')
plt.show()
