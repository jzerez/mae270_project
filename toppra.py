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
    acc_lim = 5e2

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
        _, sddot_min = calc_accel_bounds(xi, a[:, idx:idx+2], b[:, idx:idx+2], c[:, idx:idx+2], torque_lim, ds,
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
    x_max_stack = csdl.copyvar(x_max)

    x_max = csdl.minimum(x_max, axes=(0,), rho=rho)

    # x_max is the highest feasible path speed (squared). 
    # Subject to joint velocity limits and dynamic constraints
    x_max = x_max.set(csdl.slice[-1], 0)

    ### Forward Pass ###
    x = csdl.Variable(shape=(x_max.shape), value=0)
    sddot_mins = csdl.Variable(shape=(n,), value=0)
    sddot_maxs = csdl.Variable(shape=(n,), value=0)
    x_min = csdl.Variable(shape=(n,), value=0)
    for i in csdl.frange(n-1):
        xi = x[i]
        sddot_max, sddot_min = calc_accel_bounds(xi, a[:, i:i+2], b[:, i:i+2], c[:, i:i+2],
                                                 torque_lim, ds, eps=eps, rho=rho, acc_lim=acc_lim)
        sddot_mins = sddot_mins.set(csdl.slice[i], sddot_min)
        sddot_maxs = sddot_maxs.set(csdl.slice[i], sddot_max)

        x_next_min = xi + 2 * ds * sddot_min
        x_next_max = xi + 2 * ds * sddot_max

        x_min = x_min.set(csdl.slice[i], x_next_min)

        # Find the limiting path velocity between dynamics and path feasibility
        x_next1 = csdl.Variable(shape=(2,), value=0)
        x_next1 = x_next1.set(csdl.slice[0], x_max[i+1])
        x_next1 = x_next1.set(csdl.slice[1], x_next_max)
        x_next1 = csdl.minimum(x_next1, rho=rho)
        
        # Extra check, make sure we're greater than the minimum path vel for path feasibility
        # x_next2 = csdl.Variable(shape=(3,), value=0)
        # x_next2 = x_next2.set(csdl.slice[0], x_next1)
        # x_next2 = x_next2.set(csdl.slice[1], x_next_min)
        # x = x.set(csdl.slice[i+1], csdl.maximum(x_next2, rho=rho))
        x = x.set(csdl.slice[i+1], x_next1)

    ### Bring back into theta-space ###
    t = csdl.Variable(shape=(n,), value=0)
    qt = csdl.Variable(shape=(n_joints, n), value=0)
    qtt = csdl.Variable(shape=(n_joints, n), value=0)
    tau = csdl.Variable(shape=(n_joints, n), value=0)
    sdots = csdl.Variable(shape=(n,), value=0)
    sddots = csdl.Variable(shape=(n,), value=0)

    for i in csdl.frange(n - 1):
        x_i_pair = csdl.Variable(shape=(2,), value=0.0)
        x_i_pair = x_i_pair.set(csdl.slice[0], x[i])
        x_i_pair = x_i_pair.set(csdl.slice[1], 0.0)
        x_i_nonneg = csdl.maximum(x_i_pair, rho=rho)
        x_i_nonneg = x[i]

        x_ip1_pair = csdl.Variable(shape=(2,), value=0.0)
        x_ip1_pair = x_ip1_pair.set(csdl.slice[0], x[i + 1])
        x_ip1_pair = x_ip1_pair.set(csdl.slice[1], 0.0)
        x_ip1_nonneg = csdl.maximum(x_ip1_pair, rho=rho)
        x_ip1_nonneg = x[i + 1]


        sddot_max, sddot_min = calc_accel_bounds(x[i], a[:, i:i+2], b[:, i:i+2], c[:, i:i+2],
                                                 torque_lim, ds, eps=eps, rho=rho, acc_lim=acc_lim)
        
        sdot = csdl.sqrt(x_i_nonneg + eps)
        sddot = (x_ip1_nonneg - x_i_nonneg) / (2 * ds)
        sddot = csdl.minimum(csdl.vstack((sddot, sddot_max)), rho=rho)
        sddot = csdl.maximum(csdl.vstack((sddot, sddot_min)), rho=rho)

        sdots = sdots.set(csdl.slice[i], sdot)
        sddots = sddots.set(csdl.slice[i], sddot)

        dt = 2 * ds / (sdot + csdl.sqrt(x_ip1_nonneg + eps) + eps)
        t = t.set(csdl.slice[i + 1], t[i] + dt)

        qt = qt.set(csdl.slice[:, i], qdot[:, i] * sdot)
        qtt = qtt.set(csdl.slice[:, i], qdot[:, i] * sddot + qddot[:, i] * x_i_nonneg)

        tau_i = dynamics.inverse_dynamics(
            q[:, i], qt[:, i], qtt[:, i],
            link_frames, screw_axes, inertias, grav
        )
        tau = tau.set(csdl.slice[:, i], tau_i)

    return t, qt, qtt, tau, s, sdot, sddots, sddot_maxs, sddot_mins, x, x_max, x_min, x_max_stack, a, b,

def calc_x_max_joint(qdot, vel_lim, rho=200): 
    n_joints, n = qdot.shape
    x_lim = csdl.Variable(shape=(n,), value=0)
    eps = 1e-10
    for i in csdl.frange(n-1):
        x = (vel_lim / qdot[:, i] + eps)**2
        x_lim = x_lim.set(csdl.slice[i], csdl.minimum(x, rho=rho))
    return x_lim

def calc_accel_bounds(x, a, b, c, torque_lim, ds, rho=200, eps=1e-10, acc_lim=2.5e3):
    # x = sdot**2

    # NOTE: neat trick with smooth abs here
    # Sigmoid logic to deal with sign flipping. Valid because if a[i] 
    # is close to zero (where our sigmoid is not very accurate), acceleration
    # along s requires very little torque, which means that it is unlikely that 
    # that particular joint is driving the overall acceleration limit. 
    
    a = a.set(csdl.slice[:, 1], a[:, 1] + 2 * ds * b[:, 1])
    torque_lim = csdl.transpose(csdl.vstack((torque_lim, torque_lim)))
    # print(a.shape, b.shape, c.shape, torque_lim.shape)
    # u1 = (torque_lim - b * x - c) / (a + eps)
    # u2 = (-torque_lim - b * x - c) / (a + eps)

    # pair = csdl.Variable(shape=(2, a.shape[0]), value=0.0)
    # pair = pair.set(csdl.slice[0, :], u1)
    # pair = pair.set(csdl.slice[1, :], u2)

    # lower_joint = csdl.minimum(pair, axes=(0,), rho=rho)
    # upper_joint = csdl.maximum(pair, axes=(0,), rho=rho)

    # sddot_min = csdl.maximum(lower_joint, rho=rho)
    # sddot_max = csdl.minimum(upper_joint, rho=rho)

    # max_pair = csdl.Variable(shape=(2,), value=0.0)
    # max_pair = max_pair.set(csdl.slice[0], sddot_max)
    # max_pair = max_pair.set(csdl.slice[1], acc_lim)
    # sddot_max = csdl.minimum(max_pair, rho=rho)

    # min_pair = csdl.Variable(shape=(2,), value=0.0)
    # min_pair = min_pair.set(csdl.slice[0], sddot_min)
    # min_pair = min_pair.set(csdl.slice[1], -acc_lim)
    # sddot_min = csdl.maximum(min_pair, rho=rho)

    # return sddot_max, sddot_min
    upper = (torque_lim - b * x - c) / utils.abs(a, eps)

    upper_accel = csdl.Variable(shape=(upper.size + 1,), value=0)
    lower_accel = csdl.Variable(shape=(upper.size + 1,), value=0)

    upper_accel = upper_accel.set(csdl.slice[:-1], upper.flatten())
    upper_accel = upper_accel.set(csdl.slice[-1], acc_lim)

    lower_accel = lower_accel.set(csdl.slice[:-1], -upper.flatten())
    lower_accel = lower_accel.set(csdl.slice[-1], -acc_lim)

    return csdl.minimum(upper_accel, rho=rho), csdl.maximum(lower_accel, rho=rho)


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
    
    robot.inertias = robot.inertias.set(csdl.slice[-1, :, :], csdl.Variable(value=np.identity(6)*100))
    robot.inertias *= 0.5
    q_ref = csdl.Variable(value=q_ref.T)
    s_ref = csdl.Variable(value=s_ref)
    spline_fit = cubic_spline.fit_cubic_spline(s_ref, q_ref)
    s, q, qdot, qddot  = cubic_spline.discretize_spline(s_ref, q_ref, spline_fit, 200)


    grav = csdl.Variable(value=np.array([0, 0, -9.81]))*1
    torque_lim = csdl.Variable(shape=(3,), value=300)
    vel_lim = csdl.Variable(shape=(3,), value=2.2)

    t, qt, qtt, tau, s, sdot, sddot, sddot_maxes, sddot_mins, x, x_max, x_min, x_max_stack, a, b,  = toppra(
        s, q, qdot, qddot, torque_lim, vel_lim, 
        robot.link_to_link_frames, robot.screw_axes, robot.inertias, grav)
    
    
    recorder.stop()
    if True:
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
                vel_lim, x, x_max, x_min, x_max_stack, a, b
            ]
        )
        import time
        import matplotlib.pyplot as plt
        t0 = time.time()
        jax_sim.run()
        print('running time:', time.time() - t0)


        # Plot 1x3 of q(s), qs(s) and qss(s). Why the jump at the end? 
        # qs(s) has a shock at the end...
        fig, axs = plt.subplots(1, 4, figsize=(12, 4), layout="constrained",
                        gridspec_kw={'width_ratios': [1, 1, 1, 0.2]})
        
        ylabels = [r'$q$', r'$q_s$', r'$q_{ss}$']
        line_labels = ['j0', 'j1', 'j2']

        vars = [jax_sim[q], jax_sim[qdot], jax_sim[qddot]] 
        for i in range(3):
            lines = axs[i].plot(jax_sim[s], vars[i].T)
            for s_loc in jax_sim[s_ref]:
                # We only need to label the vline once for the legend to pick it up
                vl = axs[i].axvline(s_loc, color='k', linewidth=1, linestyle=':', label='Waypoints')
            
            axs[i].set(
                ylabel=ylabels[i],
                xlabel='s',
            )
            axs[i].grid()
        
        axs[3].axis('off')
        all_handles = [*lines, vl]
        all_labels = [*line_labels, 'Waypoints']

        # 3. Create the legend on the 4th axis
        axs[3].legend(all_handles, all_labels, loc='center left', frameon=False)

        fig.suptitle('Parameterized Path Information')

        # # Plot s vs t
        fig, ax = plt.subplots()
        ax.plot(jax_sim[s], jax_sim[t])
        [ax.axvline(s_loc, color='k', linewidth=1, linestyle=':', label='Waypoints') for s_loc in jax_sim[s_ref]]
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.set(
            xlabel='s', ylabel='t', title='Time Parameterization'
        )
        ax.grid()

        # plot joint torque and velocity vs. time
        fig, axs = plt.subplots(2, 4, figsize=(25, 8), gridspec_kw={'width_ratios': [1, 1, 1, 0.1]}, layout="constrained")
        mask = np.isin(jax_sim[s], jax_sim[s_ref])
        idx = np.where(mask)[0]

        for i in range(3):
            axs[0][i].plot(jax_sim[t], jax_sim[qt][i, :], label='vel')
            axs[0][i].axhline(jax_sim[vel_lim][i], linestyle='--', color='k', label='max_vel')
            axs[0][i].axhline(-jax_sim[vel_lim][i], linestyle='--', color='r', label='min_vel')
            [axs[0][i].axvline(t_loc, color='k', linestyle=':', linewidth=1, label='Waypoints') for t_loc in jax_sim[t][idx]]
            axs[0][i].set(
                title=f'J{i} Vel',
                xlabel='time [s]',
                ylabel='vel [rad/s]',
            )
            axs[0][i].grid()
           

            axs[1][i].plot(jax_sim[t], jax_sim[tau][i, :], label='torque')
            axs[1][i].axhline(jax_sim[torque_lim][i], linestyle='--', color='k', label='max_torque')
            axs[1][i].axhline(-jax_sim[torque_lim][i], linestyle='--', color='r', label='min_torque')
            [axs[1][i].axvline(t_loc, color='k', linestyle=':', linewidth=1, label='Waypoints') for t_loc in jax_sim[t][idx]]
            axs[1][i].set(
                title=f'J{i} Torque',
                xlabel='time [s]',
                ylabel='Torque [Nm]',
            )
            axs[1][i].grid()
        for i in range(2):
            legend_ax = axs[i, 3]
            legend_ax.axis('off') # Hide the axis lines/ticks
            
            # Get handles from one of the plots in the row
            handles, labels = axs[i, 0].get_legend_handles_labels()

            by_label = dict(zip(labels, handles))
            
            # Place the legend in the dedicated empty axis
            legend_ax.legend(by_label.values(), by_label.keys(), loc='center left', frameon=False)    
        fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1)

        # Plot Path accel info
        fig, ax = plt.subplots()
        ax.plot(jax_sim[s], jax_sim[sddot], label='Actual path accel')
        ax.plot(jax_sim[s], jax_sim[sddot_maxes], 'k--', label='max')
        ax.plot(jax_sim[s], jax_sim[sddot_mins], 'r--', label='min')
        [ax.axvline(s_loc, color='k', linestyle=':', linewidth=1, label='Waypoints') for s_loc in jax_sim[s_ref]]
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.set(
            xlabel='s',
            ylabel=r'$\ddot{s}$',
            title='Feasible path accels'
        )
        ax.grid()
        

        # Plot Path vel info
        fig, ax = plt.subplots()
        ax.plot(jax_sim[s], jax_sim[x], label='Actual x')
        ax.plot(jax_sim[s], jax_sim[x_max], 'k--', label=r'$\bar{x}$')
        ax.plot(jax_sim[s], jax_sim[x_min], 'r--', label=r'$-\bar{x}$')
        [ax.axvline(s_loc, color='k', linestyle=':', linewidth=1, label='Waypoints') for s_loc in jax_sim[s_ref]]
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.set(
            xlabel='s',
            ylabel=r'$\dot{s}^2$',
            title='Feasible path squared velocities'
        )
        ax.grid()

        # Plot Path vel info
        fig, ax = plt.subplots()
        ax.plot(jax_sim[s], jax_sim[x], label='Actual x')
        ax.plot(jax_sim[s], jax_sim[x_max_stack][0], 'k--', label=r'$\bar{x}^{dyn}$')
        ax.plot(jax_sim[s], jax_sim[x_max_stack][1], 'r--', label=r'$-\bar{x}^{vel}$')
        [ax.axvline(s_loc, color='k', linestyle=':', linewidth=1, label='Waypoints') for s_loc in jax_sim[s_ref]]
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        ax.set(
            xlabel='s',
            ylabel=r'$\dot{s}^2$',
            title='Feasible path squared velocities'
        )
        ax.grid()

        # Plot Path vel info
        fig, ax = plt.subplots()
        ax.plot(jax_sim[s], jax_sim[a].T, label='a')
        # ax.plot(jax_sim[s], jax_sim[x_max_stack][0], 'k--', label=r'$\bar{x}^{dyn}$')
        # ax.plot(jax_sim[s], jax_sim[x_max_stack][1], 'r--', label=r'$-\bar{x}^{vel}$')
        # [ax.axvline(s_loc, color='k', linestyle=':', linewidth=1, label='Waypoints') for s_loc in jax_sim[s_ref]]
        # handles, labels = ax.get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # ax.legend(by_label.values(), by_label.keys())
        ax.set(
            xlabel='s',
            ylabel=r'a',
            # title='Feasible path squared velocities'
        )
        ax.grid()

        # Plot Path vel info
        fig, ax = plt.subplots()
        ax.plot(jax_sim[s], jax_sim[b].T, label='b')
        # ax.plot(jax_sim[s], jax_sim[x_max_stack][0], 'k--', label=r'$\bar{x}^{dyn}$')
        # ax.plot(jax_sim[s], jax_sim[x_max_stack][1], 'r--', label=r'$-\bar{x}^{vel}$')
        # [ax.axvline(s_loc, color='k', linestyle=':', linewidth=1, label='Waypoints') for s_loc in jax_sim[s_ref]]
        # handles, labels = ax.get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # ax.legend(by_label.values(), by_label.keys())
        ax.set(
            xlabel='s',
            ylabel=r'b',
            # title='Feasible path squared velocities'
        )
        ax.grid()


        

        # plot joint torque and velocity vs. time
        fig, axs = plt.subplots(2, 2, figsize=(4, 6), gridspec_kw={'width_ratios': [1, 0.1]}, layout="constrained")
        mask = np.isin(jax_sim[s], jax_sim[s_ref])
        idx = np.where(mask)[0]

        for i in range(3):
            axs[0][0].plot(jax_sim[t], jax_sim[qt][i, :], label=f'J{i}')
            
            axs[0][0].axhline(jax_sim[vel_lim][i], linestyle='--', color='k', label='max_vel')
            axs[0][0].axhline(-jax_sim[vel_lim][i], linestyle='--', color='r', label='min_vel')
            [axs[0][0].axvline(t_loc, color='k', linestyle=':', linewidth=1, label='Waypoints') for t_loc in jax_sim[t][idx]]
            axs[0][0].set(
                title=f'Joint Velocity Profiles',
                xlabel='time [s]',
                ylabel='vel [rad/s]',
            )
            axs[0][0].grid()
           

            axs[1][0].plot(jax_sim[t], jax_sim[tau][i, :], label=f'J{i}')
            axs[1][0].axhline(jax_sim[torque_lim][i], linestyle='--', color='k', label='max_torque')
            axs[1][0].axhline(-jax_sim[torque_lim][i], linestyle='--', color='r', label='min_torque')
            [axs[1][0].axvline(t_loc, color='k', linestyle=':', linewidth=1, label='Waypoints') for t_loc in jax_sim[t][idx]]
            axs[1][0].set(
                title='Joint Torque Profiles',
                xlabel='time [s]',
                ylabel='Torque [Nm]',
            )
            axs[1][0].grid()
        for i in range(2):
            legend_ax = axs[i, 1]
            legend_ax.axis('off') # Hide the axis lines/ticks
            
            # Get handles from one of the plots in the row
            handles, labels = axs[i, 0].get_legend_handles_labels()

            by_label = dict(zip(labels, handles))
            
            # Place the legend in the dedicated empty axis
            legend_ax.legend(by_label.values(), by_label.keys(), loc='center left', frameon=False)    
        fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.1)
        plt.show(block=True)
