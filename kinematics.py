import numpy as np
import csdl_alpha as csdl
import modern_robotics as mr
from utils import *

class InverseKinematics(csdl.CustomExplicitOperation):
    def __init__(self, joint_frames, ee_frame, name):
        super().__init__()
        csdl.check_parameter(joint_frames, 'joint_frames', types=csdl.Variable)
        csdl.check_parameter(ee_frame, 'ee_frame', types=csdl.Variable)

        assert(joint_frames.size > 0)
        assert(ee_frame.size > 0)

        self.joint_frames = joint_frames.value
        self.ee_frame = ee_frame.value
        self.name = name

    def evaluate(self, inputs: csdl.VariableGroup):
        self.declare_input('theta', inputs.theta)
        self.declare_input('goal', inputs.goal)

        err = self.create_output('err', inputs.theta.shape)

        self.declare_derivative_parameters('err', 'goal', dependent=False)

        output = csdl.VariableGroup()
        output.err = err

        return output
    
    def compute(self, input_vals, output_vals):
        ## CSDL Based implementation (slow)
        # theta = csdl.Variable(value=input_vals['theta'])
        # goal = csdl.Variable(value=input_vals['goal'])

        # joint_frames = csdl.Variable(value=self.joint_frames)
        # ee_frame = csdl.Variable(value=self.ee_frame)

        # t = forward_kinematics_screw(joint_frames, theta, ee_frame)
        # err = calc_twist_err(t, goal)
        # jac = calc_jacobian(joint_frames, theta)
        # err = csdl.matmat(csdl.transpose(jac), err).value

        # output_vals['err'] = err

        ## Modern Robotics Implmentation (fast)

        assert(not self.ee_frame is None)
        assert(not self.joint_frames is None)
        
        theta = input_vals['theta']
        goal = input_vals['goal']

        t = mr.FKinSpace(self.ee_frame, self.joint_frames, theta)
        tbd = mr.TransInv(t) @ goal
        v_body = mr.se3ToVec(mr.MatrixLog6(tbd))
        v_space = mr.Adjoint(t) @ v_body
        jac = mr.JacobianSpace(self.joint_frames, theta)


        output_vals['err'] = jac.T @ v_space

    def compute_derivatives(self, inputs, outputs, derivatives):
        ## CSDL Based implementation (slow)
        # theta = csdl.Variable(value=inputs['theta'])

        # joint_frames = csdl.Variable(value=self.joint_frames)

        # jac = calc_jacobian(joint_frames, theta)
        # derivatives['err', 'theta'] = csdl.matmat(-csdl.transpose(jac), jac).value
        
        ## Modern Robotics Implementation (fast)
        theta = inputs['theta']
        jac = mr.JacobianSpace(self.joint_frames, theta)
        derivatives['err', 'theta'] = -jac.T @ jac


if __name__ == "__main__":
    import time


    s1 = np.array([0, 0, 1, 0, 0, 0])
    s2 = np.array([0, 0, 1, 0, -2, 0])
    s3 = np.array([0, 0, 1, 0, -3, 0])

    s = np.array((s1, s2, s3)).T

    angles = np.array((0.1, 0.1, 0.1))
    M = np.identity(4)
    M[0, -1] = 3
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    # Frame 1
    frame_1 = csdl.Variable(value=np.identity(4))
    cs1 = joint_transform_to_screw_axis(frame_1)

    # Frame 2
    frame_2 = csdl.Variable(value=np.identity(4))
    frame_2 = frame_2.set(csdl.slice[:3, -1], np.array((2, 0, 0)))
    cs2 = joint_transform_to_screw_axis(frame_2)

    # Frame 3 
    frame_3 = csdl.Variable(value=np.identity(4))
    frame_3 = frame_3.set(csdl.slice[:3, -1], np.array((1, 0, 0)))


    frames = csdl.Variable(shape=(4, 4, 3), value=0)
    frames = frames.set(csdl.slice[:, :, 0], frame_1)
    frames = frames.set(csdl.slice[:, :, 1], frame_2)
    frames = frames.set(csdl.slice[:, :, 2], frame_3)

    frame_3 = csdl.matmat(frame_2, frame_3)
    cs3 = joint_transform_to_screw_axis(frame_3)



    screws = csdl.Variable(shape=(6, 3), value=0)
    screws = screws.set(csdl.slice[:, 0], cs1)
    screws = screws.set(csdl.slice[:, 1], cs2)
    screws = screws.set(csdl.slice[:, 2], cs3)
    ee_frame = csdl.Variable(value=M)

    # transform = forward_kinematics(theta, frames)


    goal = np.identity(4)
    goal[:3, -1] = np.array((2.9, 0, 0))
    goal_transform = csdl.Variable(name='goal', value=goal)



    ik_input = csdl.VariableGroup()
    theta = csdl.ImplicitVariable(value=angles)
    ik_input.theta = theta
    ik_input.goal = goal

    ik = InverseKinematics(screws, ee_frame)

    ik_err = ik.evaluate(ik_input)
    solver = csdl.nonlinear_solvers.Newton()
    start = time.time()
    solver.add_state(theta, ik_err.err, tolerance=1e-3)
    solver.run()

    print('csdl ik', time.time() - start)

    recorder.stop()

    start = time.time()
    for i in range(50):
        mr.IKinSpace(s, M, goal, angles, 0.001, 0.001)
    print('mr ik', time.time() - start)



    # Evaluate: Set up the model
    #  
    #
    # Compute: 
    # return the error
    # 
    # Compute Derivative 
    # Return the derivative of the error twist (pseudo-inverse of Jacobian)? 
            