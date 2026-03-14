import numpy as np
import csdl_alpha as csdl
import utils
import kinematics
import mass_inertia
import actuator_model


class Robot():
    def __init__(self, j_ang, j_pos, 
                 a_power, a_gear, 
                 l_id, l_t):
        """Initialize a Robot object

        Args:
            j_ang (csdl.Variable): (n_joints, 3) Euler angles of each joint's axis relative to its parent [rad]
                first joint is expressed relative to the global/world frame.
            j_pos (csdl.Variable): (n_joints, 3) Location of each joint's frame relative to its parent [m]
                first joint is expressed relative to the global/world frame and should be zero
            a_power (csdl.Variable): (n_joints,) Power for each actuator [W]
            a_gear (csdl.Variable): (n_joints,) Gear ratio for each actuator. Larger value = more torque[unitless]
            l_id (csdl.Variable): (n_links,) Inner diameter for each link [m]
            l_t (csdl.Variable): (n_links,) Wall-thickness for each link [m]

        Attributes:
            n_joints (int): number of joints
            joint_frames (csdl.Variable): (n_joints, 4, 4) transforms for each joint relative to its parent. 
                Z-axis defines rotation axis
            screw_axes (csdl.Variable): (n_joints, 6) screw axes for each actuator when the robot is in 
                the zero config. Defined in the world frame
            a_masses (csdl.Variable): (n_joints,) mass of each actuator [m]
            a_power (csdl.Variable): (n_joints,) power of each actuator [W]
            a_gear (csdl.Variable): (n_joints,) gear ratio of each actuator [unitless]
            torque_lims (csdl.Variable): (n_joints,) max torque of each actuator. Applied in positive and negative direction. [Nm]
            vel_lims (csdl.Variable): (n_joints,) max velocity of each actuator. Applied in positive and netavie direction. [rad/s]
            rho (float): density of link material. Defaults to 2700 for Aluminum [kg/m^3]
            link_lens (csdl.Variable): (n_links,) length of each link [m]
            link_masses (csdl.Variable): (n_links,) mass of each link [kg]
            inertias (csdl.Variable): (n_links, 6, 6) Spatial inertia for each link [kg, kgm^2]
            link_to_joint_frames (csdl.Variable): (n_links, 4, 4) transform expressing each link's COM frame
                relative to the joint frame of the nearest-upstream actuator (ie: the actuator that moves the link)
            link_to_link_frames (csdl.Variable): (n_links, 4, 4) transform expressing each link's COM frame
                relative to the link COM frame of the nearest-upstream link 
            total_mass (csdl.Variable): (1,) total robot mass [m]
        Notes:
            - Joint frames are defined with Z-axis representing rotation
            - Link frames are defined at the link's center of mass and with axes aligned with the principal moments of inertia
            - A robot always has n_links = n_joints + 1
                - 1st link: body of the 1st actuator
                    - Assume the link's COM frame is co-located with the joint's frame
                - 2nd link: body of the 2nd actuator + body of 1st tube
                - ith link: body of the ith actuator + body of (i-1)th tube
                - nth link: body of the end-effector (what is attached to the last joint)
            - A robot's end effector is assumed to have mass equal to the payload and be attached directly to the frame of the last joint
            - A robot's first joint must be located at the origin but may have any orientation                 

        """

        # todo: add param checks
        self.n_joints = j_ang.shape[0]

        # todo: make variable group to store inputs to init(). good for future debug

        self.joint_frames = self.calc_joint_frames(j_ang, j_pos)
        self.screw_axes = self.calc_screw_axes()
        self.a_masses, self.torque_lims, self.vel_lims = self.calc_actuator_params(a_power, a_gear)
        
        self.a_power = a_power
        self.a_gear = a_gear

        rho = 2700 # [kg/m^3]

        payload=5 # [kg]

        self.link_lens = self.calc_link_lens()
        
        # Shift actuator masses down for link inertia calcs. Assumes EOAT is zero mass.
        a_masses_temp = csdl.Variable(shape=self.a_masses.shape, value=1e-10)
        for i in csdl.frange(self.n_joints - 1):
            a_masses_temp = a_masses_temp.set(csdl.slice[i], self.a_masses[i+1])

        self.link_masses, coms, self.inertias = mass_inertia.build_Glist(self.link_lens, l_id, l_t, rho, a_masses_temp)
        self.link_to_joint_frames, self.link_to_link_frames = self.calc_link_frames(coms)
        # pass

    def calc_joint_frames(self, j_ang, j_pos):

        frames = csdl.Variable(shape=(self.n_joints, 4, 4), value=0)

        for i in csdl.frange(self.n_joints):
            rot = utils.euler_angle_to_rotation_matrix(j_ang[i, :])
            frame = csdl.Variable(value=np.identity(4))
            frame = frame.set(csdl.slice[:3, :3], rot)
            frame = frame.set(csdl.slice[:3, -1], j_pos[i, :])
            frames = frames.set(csdl.slice[i, :, :], frame)

        return frames
    
    def calc_screw_axes(self):
        screws = csdl.Variable(shape=(self.n_joints, 6), value=0)
        world_transform = csdl.Variable(value=np.identity(4))

        for i in csdl.frange(self.n_joints):
            world_transform = csdl.matmat(world_transform, self.joint_frames[i])
            screw = utils.joint_transform_to_screw_axis(world_transform)
            screws = screws.set(csdl.slice[i], screw)
        
        return screws
    
    def calc_actuator_params(self, a_power, a_gear):
        masses = csdl.Variable(shape=(self.n_joints,), value=0)
        torque_lims = csdl.Variable(shape=(self.n_joints,), value=0)
        vel_lims = csdl.Variable(shape=(self.n_joints,), value=0)

        for i in csdl.frange(self.n_joints):
            
            mass = actuator_model.actuator_mass_from_power(a_power[i])
            torque_lim, vel_lim, _ = actuator_model.actuator_limits_from_power_and_gear(a_power[i], a_gear[i])
            
            masses = masses.set(csdl.slice[i], mass)
            torque_lims = torque_lims.set(csdl.slice[i], torque_lim)
            vel_lims = vel_lims.set(csdl.slice[i], vel_lim)
        return masses, torque_lims, vel_lims
    
    def calc_link_frames(self, coms):
        # transform of frame i, relative to frame i-1
        # Frames are defined at each link's COM
        # Convention is that x-axis is colinear with link axis 
        n_links = self.n_joints + 1

        i4 = csdl.Variable(value=np.identity(4))

        # Define a link's COM frame relative to the joint frame of the nearest-upstream actuator 
        # (ie: the actuator that moves the current link)
        link_to_joint_frames = csdl.Variable(shape=(n_links, 4, 4), value=0)

        # Define a link's COM frame relative to the joint frame of the nearest-upstream link
        link_to_link_frames = csdl.Variable(shape=(n_links, 4, 4), value=0)

        np.random.seed(101)
        noise = csdl.Variable(value=np.random.random((3,)))

        for i in csdl.frange(1, n_links-2):
            frame = csdl.Variable(value=np.identity(4))
            # Frame of the actuator at the end of the current link
            child_joint_frame = self.joint_frames[i+1, :, :]

            # Choose coordinate system for link. y and z are arbitrary.
            x_vec = utils.unit(child_joint_frame[:3, -1])
            y_vec = csdl.cross(utils.unit(x_vec + noise), x_vec)
            z_vec = csdl.cross(x_vec, y_vec)

            com = coms[i, :]
            
            frame = frame.set(csdl.slice[:3, 0], x_vec)
            frame = frame.set(csdl.slice[:3, 1], y_vec)
            frame = frame.set(csdl.slice[:3, 2], z_vec)
            frame = frame.set(csdl.slice[:3, 3], com)
            
            link_to_joint_frames = link_to_joint_frames.set(csdl.slice[i], frame)

        

        # NOTE: this assumes EOAT is co-located to last joint. 
        link_to_joint_frames = link_to_joint_frames.set(csdl.slice[-1, :, :], i4)
        # This assumes that the first link (the first actuator body) is aligned with the first actuator's frame. 
        link_to_joint_frames = link_to_joint_frames.set(csdl.slice[0, :, :], i4)

        link_to_link_frames = link_to_link_frames.set(csdl.slice[0, :, :], i4)
        for i in csdl.frange(1, n_links-1):
            t_lj = utils.invert_transform(link_to_joint_frames[i, :, :])
            t_jj = self.joint_frames[i, :, :]
            t_jl = link_to_joint_frames[i, :, :]

            t_ll = csdl.matmat(t_lj, csdl.matmat(t_jj, t_jl))

            link_to_link_frames = link_to_link_frames.set(csdl.slice[i, :, :], t_ll)
        
        return link_to_joint_frames, link_to_link_frames

    def calc_link_lens(self):
        lens = csdl.Variable(shape=(self.n_joints, ), value=0)
        for i in csdl.frange(1, self.n_joints):
            p = self.joint_frames[i, :3, -1]
            lens = lens.set(csdl.slice[i-1], csdl.norm(p))
        return lens




if __name__ == "__main__":
    rec = csdl.Recorder(inline=True)
    rec.start()
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
    l_id = csdl.Variable(value=np.array([0, 0.2, 0.2, 0]))
    l_t = csdl.Variable(value=np.array([0, 0.01, 0.01, 0]))

    print(j_ang.shape)


    robot = Robot(j_ang, j_pos, 
                 a_power, a_gear, 
                 l_id, l_t)
    
    rec.stop()

    print(robot.link_lens.value)
    print(robot.link_masses.value)

    print(robot.inertias.value)


    
