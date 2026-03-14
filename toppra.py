import csdl_alpha as csdl
import numpy as np
import dynamics
import utils
import cubic_spline
from Robot import Robot

def toppra(s, q, qdot, qddot, torque_lim, vel_lim,
           link_frames, screw_axes, inertias, grav):
    """_summary_

    Args:
        s (csdl.Variable): (n,) Vector of parameterizer values
        q (csdl.Variable): (n_joints, n) parameterized path [rad]
        qdot (csdl.Variable): (n_joints, n) parameterized path velocity dq/ds [rad/unit]
        qddot (csdl.Variable): (n_joints, n) parameterized path accel d^2q/ds^2 [rad/unit^2] 
        torque_lim (csdl.Variable): (n_joints) torque limits. Applied in positive and negative directions
        vel_lim (csdl.Variable): (n_joints) vel limits. Applied in positive and negative directions
        link_frames (csdl.Variable): (n_links, 4, 4) transform of link {i}, relative to upstream link {i-1}
        screw_axes (csdl.Variable): (n_joints, 6) Screw axes for robot's joints in the zero config
        inertias (csdl.Variable): (n_links, 6, 6) Inertia matrices for each link of robot
        grav (csdl.Variable): (3,) Gravity vector [0, 0, -9.81] represents z+ is upward. [m/s/s]

    Returns:
        t (csdl.Variable): (n,) time at each s
        qt (csdl.Variable): (n_joints, n) joint velocity [rad/s]
        qtt (csdl.Variable): (n_joints, n) joint acceleration [rad/s/s]
        tau (csdl.Variable): (n_joints, n) joint torques [Nm]
        s (csdl.Variable): (n,) parametric coords [unit]
        sdot (csdl.Variable): (n,) parametric velocity 
        sddots (csdl.Variable): (n,) parametric accel
        sddot_maxs (csdl.Variable): (n,) max parametric accel
        sddot_mins (csdl.Variable): (n,) min parametric accel

    """
    n_joints = q.shape[0]
    n = s.shape[0]

    # Delta s. This is constant
    ds = s[1] - s[0]

    # Acceleration limit
    acc_lim = 5e3

    eps = 1e-10

    # param for smoothmax/smoothmins
    rho = 200    

    ### Dynamics equations in parametric space ###
    # Cannonical dynamics equation is:
    #   Tau = M(q)q_tt + C(q, q_t) + g(q)
    # We want to represent in the parametrized space, s:
    #   Tau = a(s)s_tt + b(s)s_t + c(s)
    # 
    # Here, a(s) = M(q)q_s, b = M(q)q_ss + C(q, q_s)q_s, c = g(q)
    a = csdl.Variable(shape=(n_joints, n), value=0)
    b = csdl.Variable(shape=(n_joints, n), value=0)
    c = csdl.Variable(shape=(n_joints, n), value=0)

    zero = csdl.Variable(shape=(n_joints, ), value=0)
    for i in csdl.frange(n):
        ci = dynamics.inverse_dynamics(q[:, i], zero, zero, link_frames, screw_axes, inertias, grav)
        c = c.set(csdl.slice[:, i], ci)

        ai = dynamics.inverse_dynamics(q[:, i], zero, qdot[:, i], link_frames, screw_axes, inertias, grav)
        ai -= ci

        a = a.set(csdl.slice[:, i], ai)

        bi = dynamics.inverse_dynamics(q[:, i], qdot[:, i], qddot[:, i], link_frames, screw_axes, inertias, grav)
        bi -= ci
        b = b.set(csdl.slice[:, i], bi)

    
    ### Backwards pass ###
    # max path velocity, determined by joint velocity limits. x = sdot^2
    x_max_joint = calc_x_max_joint(qdot, vel_lim, rho)

    # max path velocity, determined by dynamics
    x_max_dyn = csdl.Variable(shape=x_max_joint.shape, value=0)

    for i in csdl.frange(n-1):
        # Iterate backwards
        idx = n - i - 2

        xi = csdl.ImplicitVariable(shape=(1,), value=100)

        # Find best xi that satisfies dynamic constraints
        _, sddot_min = calc_accel_bounds(xi, a[:, idx], b[:, idx], c[:, idx], torque_lim,
                                         eps=eps, rho=rho, acc_lim=acc_lim)
        
        # Guarantee we're smaller than x_max to within convergence tolerance 
        tol = 1e-10
        residual = xi + 2 * ds * sddot_min - (x_max_dyn[idx+1] * 0.999)

        solver = csdl.nonlinear_solvers.Newton(print_status=False)
        solver.add_state(xi, residual)
        solver.run()
        
        x_max_dyn = x_max_dyn.set(csdl.slice[idx], xi)

    # Find soft-min of path-velocity
    x_max = csdl.Variable(shape=(2, n), value=0)
    x_max = x_max.set(csdl.slice[0, :], x_max_joint)
    x_max = x_max.set(csdl.slice[1, :], x_max_dyn)

    x_max = csdl.minimum(x_max, axes=(0,), rho=rho)

    # x_max is the highest feasible path speed (squared). 
    # Subject to joint velocity limits and dynamic constraints
    x_max = x_max.set(csdl.slice[-1], 0)

    ### Forward Pass ###
    x = csdl.Variable(shape=(x_max.shape), value=0)
    sddot_mins = csdl.Variable(shape=(n,), value=0)
    sddot_maxs = csdl.Variable(shape=(n,), value=0)
    for i in csdl.frange(n-1):
        xi = x[i]
        sddot_max, sddot_min = calc_accel_bounds(xi, a[:, i], b[:, i], c[:, i], 
                                                 torque_lim, eps=eps, rho=rho, acc_lim=acc_lim)
        sddot_mins = sddot_mins.set(csdl.slice[i], sddot_min)
        sddot_maxs = sddot_maxs.set(csdl.slice[i], sddot_max)

        x_next_min = xi + 2 * ds * sddot_min
        x_next_max = xi + 2 * ds * sddot_max

        # Find the limiting path velocity between dynamics and path feasibility
        x_next1 = csdl.Variable(shape=(2,), value=0)
        x_next1 = x_next1.set(csdl.slice[0], x_max[i+1])
        x_next1 = x_next1.set(csdl.slice[1], x_next_max)
        x_next1 = csdl.minimum(x_next1, rho=rho)
        
        # Extra check, make sure we're greater than the minimum path vel for path feasibility
        x_next2 = csdl.Variable(shape=(2,), value=0)
        x_next2 = x_next2.set(csdl.slice[0], x_next1)
        x_next2 = x_next2.set(csdl.slice[1], x_next_min)

        x = x.set(csdl.slice[i+1], csdl.maximum(x_next2, rho=rho))

    ### Bring back into theta-space ###
    t = csdl.Variable(shape=(n,), value=0)
    qt = csdl.Variable(shape=(n_joints, n), value=0)
    qtt = csdl.Variable(shape=(n_joints, n), value=0)
    tau = csdl.Variable(shape=(n_joints, n), value=0)
    sdots = csdl.Variable(shape=(n,), value=0)
    sddots = csdl.Variable(shape=(n,), value=0)

    for i in csdl.frange(n-1):
        sdot = csdl.sqrt(x[i])
        sddot = (x[i+1] - x[i]) / (2*ds)

        sdots = sdots.set(csdl.slice[i], sdot)    
        sddots = sddots.set(csdl.slice[i], sddot)  

        dt = 2*ds / (sdot + csdl.sqrt(x[i+1]) + eps)
        t = t.set(csdl.slice[i+1], t[i] + dt)
        
        qt = qt.set(csdl.slice[:, i], qdot[:, i] * sdot)
        qtt = qtt.set(csdl.slice[:, i], qdot[:, i] * sddot + qddot[:, i] * x[i])
        
        tau_i = dynamics.inverse_dynamics(q[:, i], qt[:, i], qtt[:, i], link_frames, screw_axes, inertias, grav)
        tau = tau.set(csdl.slice[:, i], tau_i)

    return t, qt, qtt, tau, s, sdot, sddots, sddot_maxs, sddot_mins, x_max

def calc_x_max_joint(qdot, vel_lim, rho=200): 
    n_joints, n = qdot.shape
    x_lim = csdl.Variable(shape=(n,), value=0)
    eps = 1e-10
    for i in csdl.frange(n-1):
        x = (vel_lim / qdot[:, i] + eps)**2
        x_lim = x_lim.set(csdl.slice[i], csdl.minimum(x, rho=rho))
    return x_lim

def calc_accel_bounds(x, a, b, c, torque_lim, rho=200, eps=1e-10, acc_lim=2.5e3):
    # x = sdot**2

    # NOTE: neat trick with smooth abs here
    # Sigmoid logic to deal with sign flipping. Valid because if a[i] 
    # is close to zero (where our sigmoid is not very accurate), acceleration
    # along s requires very little torque, which means that it is unlikely that 
    # that particular joint is driving the overall acceleration limit. 
    upper = (torque_lim - b * x - c) / utils.smooth_abs(a + eps, k=10)

    amax = csdl.minimum(upper, rho=rho)
    amin = csdl.maximum(-upper, rho=rho)

    clamped_amax = csdl.Variable(shape=(2,), value=0)
    clamped_amin = csdl.Variable(shape=(2,), value=0)

    clamped_amax = clamped_amax.set(csdl.slice[0], amax)
    clamped_amax = clamped_amax.set(csdl.slice[1], acc_lim)

    clamped_amin = clamped_amin.set(csdl.slice[0], amin)
    clamped_amin = clamped_amin.set(csdl.slice[1], -acc_lim)

    return csdl.minimum(clamped_amax, rho=rho), csdl.maximum(clamped_amin, rho=rho)


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

    use_inline=False
    recorder = csdl.Recorder(inline=use_inline)
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
    s, q, qdot, qddot  = cubic_spline.discretize_spline(s_ref, q_ref, spline_fit, 50)


    grav = csdl.Variable(value=np.array([0, 0, -9.81]))
    torque_lim = csdl.Variable(shape=(3,), value=800)
    vel_lim = csdl.Variable(shape=(3,), value=10)

    t, qt, qtt, tau, s, sdot, sddot, sddot_maxes, sddot_mins, x_max = toppra(
        s, q, qdot, qddot, torque_lim, vel_lim, 
        robot.link_to_link_frames, robot.screw_axes, robot.inertias, grav)
    
    
    recorder.stop()
    if not use_inline:
        jax_sim = csdl.experimental.JaxSimulator(
            recorder=recorder,
            additional_inputs= [
                j1_pos,
                j1_ang,
                j2_pos,
                j2_ang,
                j3_pos,
                j3_ang,
                j_ang,
                j_pos,
                a_power,
                a_gear,
                l_id,
                l_t,
                q_ref,
                s_ref,
            ],
            additional_outputs=[
                t,
                qt,
                qtt,
                tau,
                q,
                qdot,
                qddot,
                s,
                sdot,
                sddot,
                sddot_maxes,
                sddot_mins,
                torque_lim,
                vel_lim,
                x_max,
            ]
        )
        import time
        import matplotlib.pyplot as plt
        t0 = time.time()
        jax_sim.run()
        print('running time:', time.time() - t0)


        # Plot 1x3 of q(s), qs(s) and qss(s). Why the jump at the end? 
        # qs(s) has a shock at the end...
        fig, axs = plt.subplots(1, 3)
        titles = ['q', 'qs', 'qss']
        vars = [jax_sim[q], jax_sim[qdot], jax_sim[qddot]] 
        for i in range(3):
            axs[i].plot(jax_sim[s], vars[i].T)
            [axs[i].axvline(s_loc, color='k', linestyle=':', linewidth=1) for s_loc in jax_sim[s_ref]]
            axs[i].set_title(titles[i])
            axs[i].grid()

        # # Plot s vs t
        fig, ax = plt.subplots()
        ax.plot(jax_sim[s], jax_sim[t])
        ax.set(
            xlabel='s', ylabel='t'
        )



        # print('s:\n', jax_sim[s], '\nt:\n', jax_sim[t], '\n x \n:', jax_sim[x])

        # print('--------------')
        # ds = jax_sim[s]
        # x_check = jax_sim[x] + 2*0.004*jax_sim[sddot]
        # print('dynamic consistency;')
        # print(jax_sim[x][1:] - x_check[:-1])


        # fig, ax = plt.subplots()
        # ax.plot(jax_sim[s], jax_sim[all_x][1, :], label='x max')
        # ax.plot(jax_sim[s], jax_sim[x], label='x')
        # ax.plot(jax_sim[s], jax_sim[all_x][0, :], label='x lim')
        # [ax.axvline(s_loc, color='k', linestyle=':', linewidth=1) for s_loc in jax_sim[s_ref]]
        # ax.set(
        #     xlabel='s',
        #     ylabel='x'
        # )
        # ax.legend()


        # Plot 1x1 of tau vs. a*sddot + b*x + c
        # They match very well. 
        # fig, ax = plt.subplots()
        # ax.plot(jax_sim[t], jax_sim[tau].T, label='ID torque')
        # ax.plot(jax_sim[t], (jax_sim[a] * jax_sim[sddot] + jax_sim[b] * jax_sim[sdot]**2 + jax_sim[c]).T, 'k--', alpha=0.8, label='abc torque')
        # # [ax.axvline(s_loc, color='k', linestyle=':', linewidth=1) for s_loc in jax_sim[s_ref]]
        # ax.legend()
        # ax.set_title('ID torque vs. abc torque')

        # Plot 1x1 of sddot vs sddot limits 
        # At each waypoint, it seems like there is a spike in sddot bandwidth. 
        # At the third waypoint, the max and min lines cross(!)
        # The last waypoint explodes off to a super high value...
        fig, ax = plt.subplots()
        ax.plot(jax_sim[s], jax_sim[sddot])
        ax.plot(jax_sim[s], jax_sim[sddot_maxes], 'k--', label='sddot_max', alpha=0.5)
        ax.plot(jax_sim[s], jax_sim[sddot_mins], 'r:', label='sddot_min', alpha=0.5)        
        [ax.axvline(s_loc, color='k', linestyle=':', linewidth=1, label='waypt') for s_loc in jax_sim[s_ref]]
        # [ax.axvline(s_loc, color='k', linewidth=1, label='weird') for s_loc in jax_sim[s][idx]]
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        # Maybe turn off the secondary 

        # q_ref = np.array([
        #     [0.0, 0.0, 0.25],
        #     [0.5, 1.5, 0.0],
        #     [1.2, 0.5, 3.0],
        #     [1.0, 2.0, 2.0],
        #     [2.0, 0.0, 2.0],
        #     [2.0, 0.25, 1.5]
        # ])
        # fig, axs = plt.subplots(1, 2)
        # ax = axs[0]
        # ax.plot(jax_sim[s], jax_sim[all_x][0, :])
        # [ax.axvline(l, linestyle='--', color='k') for l in jax_sim[s_ref]]        
        # ax.set_title('x_dot limited by joint lims')
        # ax.set_yscale('log')

        # ax = axs[1]
        # ax.plot(jax_sim[s], jax_sim[all_x][1, :])
        # [ax.axvline(l, linestyle='--', color='k') for l in jax_sim[s_ref]]  
        # ax.set_title('x_dot limited by dyn')
        # ax.set_yscale('log')

        # jax_xmax = jax_sim[all_x][1, :]
        # weird = jax_xmax[np.bitwise_not(jax_xmax > 5e7)]

        # fig, ax = plt.subplots()
        # ax.plot(jax_sim[s], jax_sim[qddot][0, :], label='j0')
        # ax.plot(jax_sim[s], jax_sim[qddot][1, :], label='j1')
        # ax.plot(jax_sim[s], jax_sim[qddot][2, :], label='j2')
        # ax.legend()
        # ax.set_title('qddot')
        fig, axs = plt.subplots(2, 3, figsize=(20, 8))
        
        for i in range(3):
            axs[0][i].plot(jax_sim[t], jax_sim[qt][i, :], label='vel')
            axs[0][i].axhline(jax_sim[vel_lim][i], linestyle='--', color='k', label='max_vel')
            axs[0][i].axhline(-jax_sim[vel_lim][i], linestyle='--', color='r', label='min_vel')
            axs[0][i].set(
                title=f'J{i} Vel',
                xlabel='time [s]',
                ylabel='vel [rad/s]',
            )
            axs[0][i].grid()
            axs[0][i].legend()

            axs[1][i].plot(jax_sim[t], jax_sim[tau][i, :])
            axs[1][i].axhline(jax_sim[torque_lim][i], linestyle='--', color='k', label='max_torque')
            axs[1][i].axhline(-jax_sim[torque_lim][i], linestyle='--', color='r', label='min_torque')

            axs[1][i].set(
                title=f'J{i} Torque',
                xlabel='time [s]',
                ylabel='Torque [Nm]',
            )
            axs[1][i].grid()
            axs[1][i].legend()

        fig, ax = plt.subplots()
        ax.plot(jax_sim[s], jax_sim[sddot])
        ax.plot(jax_sim[s], jax_sim[sddot_maxes], 'k--', label='max')
        ax.plot(jax_sim[s], jax_sim[sddot_mins], 'r--', label='min')
        ax.plot(jax_sim[s], jax_sim[x_max], 'g-')
        ax.set(
            xlabel='s',
            ylabel=r'$\dot{\dot{s}}$',
            title='Feasible path accels'
        )
        ax.grid()
        plt.show(block=True)

