import numpy as np
import csdl_alpha as csdl


def outer_radius(r_inner, thickness):
    return r_inner + thickness


def parallel_axis(I_com, mass, d):
    """
    Parallel-axis theorem:
        I_new = I_com + m (||d||^2 I - d d^T)

    Parameters
    ----------
    I_com : csdl.Variable, shape (3,3)
    mass : csdl.Variable or scalar
    d : csdl.Variable, shape (3,)

    Returns
    -------
    csdl.Variable, shape (3,3)
    """
    I3 = csdl.Variable(value=np.eye(3))
    d_sq = csdl.vdot(d, d)
    ddT = csdl.outer(d, d)
    return I_com + mass * (d_sq * I3 - ddT)


def hollow_tube_properties(length, r_inner, thickness, density):
    """
    Hollow circular tube aligned with local x-axis.

    Parameters
    ----------
    length : csdl.Variable or scalar
    r_inner : csdl.Variable or scalar
    thickness : csdl.Variable or scalar
    density : csdl.Variable or scalar

    Returns
    -------
    mass : csdl.Variable
    com : csdl.Variable, shape (3,)
    I_com : csdl.Variable, shape (3,3)
    """
    r_outer = outer_radius(r_inner, thickness)

    volume = np.pi * length * (r_outer**2 - r_inner**2)
    mass = density * volume

    # center of mass at midpoint along x-axis
    com = csdl.Variable(shape=(3,), value=0.0)
    com = com.set(csdl.slice[0], 0.5 * length)

    # inertia tensor of hollow cylinder about CoM, axis along x
    Ixx = 0.5 * mass * (r_outer**2 + r_inner**2)
    Iyy = (1.0 / 12.0) * mass * (3.0 * (r_outer**2 + r_inner**2) + length**2)
    Izz = Iyy

    I_com = csdl.Variable(shape=(3, 3), value=0.0)
    I_com = I_com.set(csdl.slice[0, 0], Ixx)
    I_com = I_com.set(csdl.slice[1, 1], Iyy)
    I_com = I_com.set(csdl.slice[2, 2], Izz)

    return mass, com, I_com


def link_inertial_properties(length,
                             r_inner,
                             thickness,
                             density,
                             motor_mass=None,
                             motor_pos=None):
    """
    Combine a hollow-tube link with a lumped motor mass.

    Parameters
    ----------
    length, r_inner, thickness, density : csdl.Variable or scalar
    motor_mass : csdl.Variable or scalar, optional
    motor_pos : csdl.Variable, shape (3,), optional

    Returns
    -------
    mass_total : csdl.Variable
    com_total : csdl.Variable, shape (3,)
    I_total_com : csdl.Variable, shape (3,3)
    """
    mass_tube, com_tube, I_tube = hollow_tube_properties(
        length, r_inner, thickness, density
    )

    if motor_mass is None:
        motor_mass = csdl.Variable(value=0.0)

    if motor_pos is None:
        motor_pos = csdl.Variable(shape=(3,), value=np.zeros(3))

    mass_total = mass_tube + motor_mass

    # weighted average CoM
    com_total = (mass_tube * com_tube + motor_mass * motor_pos) / mass_total

    # shift tube inertia to total CoM
    d_tube = com_tube - com_total
    I_total = parallel_axis(I_tube, mass_tube, d_tube)

    # add motor as point mass
    d_motor = motor_pos - com_total
    I3 = csdl.Variable(value=np.eye(3))
    I_motor = motor_mass * (
        csdl.vdot(d_motor, d_motor) * I3 - csdl.outer(d_motor, d_motor)
    )

    I_total = I_total + I_motor

    return mass_total, com_total, I_total


def spatial_inertia(mass, I_com):
    """
    Build 6x6 spatial inertia matrix in the CoM frame:
        G = [ I_com   0
              0      mI ]

    Parameters
    ----------
    mass : csdl.Variable or scalar
    I_com : csdl.Variable, shape (3,3)

    Returns
    -------
    G : csdl.Variable, shape (6,6)
    """
    G = csdl.Variable(shape=(6, 6), value=0.0)
    G = G.set(csdl.slice[:3, :3], I_com)
    G = G.set(csdl.slice[3:, 3:], mass * csdl.Variable(value=np.eye(3)))
    return G


def build_Glist(link_params):
    """
    Build a list of spatial inertias for a robot.

    Parameters
    ----------
    link_params : list of dict
        Each dict should contain:
            {
                "L": link length,
                "ri": inner radius,
                "t": wall thickness,
                "rho": density,
                "motor_mass": motor mass (optional),
                "motor_pos": motor position, np.array shape (3,) (optional)
            }

    Returns
    -------
    data : dict
        {
            "masses": [...],
            "coms": [...],
            "inertias": [...],
            "Glist": [...]
        }
    """
    masses = []
    coms = []
    inertias = []
    Glist = []

    for p in link_params:
        motor_mass = p.get("motor_mass", None)
        motor_pos = p.get("motor_pos", None)

        if motor_pos is not None and not isinstance(motor_pos, csdl.Variable):
            motor_pos = csdl.Variable(shape=(3,), value=np.array(motor_pos))

        m, c, I = link_inertial_properties(
            p["L"],
            p["ri"],
            p["t"],
            p["rho"],
            motor_mass=motor_mass,
            motor_pos=motor_pos,
        )

        G = spatial_inertia(m, I)

        masses.append(m)
        coms.append(c)
        inertias.append(I)
        Glist.append(G)

    return {
        "masses": masses,
        "coms": coms,
        "inertias": inertias,
        "Glist": Glist,
    }


if __name__ == "__main__":
    import csdl_alpha as csdl
    import numpy as np


    recorder = csdl.Recorder(inline=True)
    recorder.start()

    L = csdl.Variable(value=0.4)
    ri = csdl.Variable(value=0.01)
    t = csdl.Variable(value=0.002)
    rho = csdl.Variable(value=2700.0)

    motor_mass = csdl.Variable(value=0.5)
    motor_pos = csdl.Variable(shape=(3,), value=np.array([0., 0., 0.]))

    m, c, I = link_inertial_properties(L, ri, t, rho, motor_mass, motor_pos)
    G = spatial_inertia(m, I)

    recorder.stop()

    print("mass =", m.value)
    print("com =", c.value)
    print("inertia =\n", I.value)
    print("spatial inertia =\n", G.value)
    import csdl_alpha as cs
    import numpy as np
    from mass_inertia import *

    recorder = cs.Recorder(inline=True)
    recorder.start()

    link_params = [
        {
            "L": cs.Variable(value=0.4),
            "ri": cs.Variable(value=0.01),
            "t": cs.Variable(value=0.002),
            "rho": cs.Variable(value=2700.0),
            "motor_mass": cs.Variable(value=0.5),
            "motor_pos": np.array([0.0, 0.0, 0.0]),
        },
        {
            "L": cs.Variable(value=0.3),
            "ri": cs.Variable(value=0.009),
            "t": cs.Variable(value=0.002),
            "rho": cs.Variable(value=2700.0),
            "motor_mass": cs.Variable(value=0.4),
            "motor_pos": np.array([0.0, 0.0, 0.0]),
        },
    ]

    data = build_Glist(link_params)

    recorder.stop()

    print("m1 =", data["masses"][0].value)
    print("m2 =", data["masses"][1].value)
    print("G1 =\n", data["Glist"][0].value)
    print("G2 =\n", data["Glist"][1].value)