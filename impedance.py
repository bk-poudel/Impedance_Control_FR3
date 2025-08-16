import numpy as np
import matplotlib.pyplot as plt
import time
from MujocoSim import FR3Sim
from scipy.spatial.transform import Rotation as R
my_robot = FR3Sim()
# Get initial end-effector pose and position
q_init, dq_init = my_robot.get_state()
T_init = my_robot.get_pose(q_init)
print("Initial pose:", T_init)
x_init = T_init[:3, 3]
R_init = T_init[:3, :3]
# Impedance parameters (6x6)
K_imp = np.diag([400.0, 400.0, 400.0, 40.0, 40.0, 40.0])   # Stiffness (N/m, Nm/rad)
D_imp = np.diag([60.0, 60.0, 60.0, 6.0, 6.0, 6.0])         # Damping (N·s/m, Nm·s/rad)
dt = 0.001
steps = 50000
for i in range(steps):
    q, dq = my_robot.get_state()
    T = my_robot.get_pose(q)
    x = T[:3, 3]
    R_ee = T[:3, :3]
    J = my_robot.get_jacobian(q)
    J_spatial = np.block([
        [R_ee @ J[3:, :]],  # Linear part
        [R_ee @ J[:3, :]]   # Angular part
    ])
    dx = J_spatial @ dq  # [vx, vy, vz, wx, wy, wz]
    # Position error
    pos_err = x - x_init
    # Orientation error (rotation vector)
    R_err = R_ee @ R_init.T
    rotvec_err = R.from_matrix(R_err).as_rotvec()
    # Stack errors
    err = np.concatenate([pos_err, rotvec_err])
    derr = dx  # [linear_vel, angular_vel]
    # Spring-damper wrench
    wrench = -K_imp @ err - D_imp @ derr
    tau = J_spatial.T @ wrench + my_robot.get_gravity(q)
    my_robot.send_joint_torque(tau, 10)  # Keep gripper open
    time.sleep(dt)
