import numpy as np
from scipy.constants import g
from pybullet_utils.transformations import quaternion_slerp, quaternion_multiply, quaternion_conjugate
import utils
from casadi import *
import json

import casadi as ca
import os
import matplotlib.pyplot as plt
# 打拍子动作
# 足端位置（在世界系下相对于质心的位置）[13:25][go2:[FR, FL, RR, RL]]
#  动作时间2s


num_row = 300
fps = 50
h = 0.08
a1 = 0.01
a2 = 0.01
T = 1

"""
生成打拍子动作轨迹并导出到文件。

Args:
    num_row (int): 轨迹帧数。
    fps (int): 每秒帧数。
    h (float): Z 方向的运动幅度。
    a1 (float): 正向 X 方向运动幅度。
    a2 (float): 负向 X 方向运动幅度。
    T (float): 动作周期。
    outfile (str): 输出文件路径。
"""
num_col = 49
beat_ref = np.ones((num_row-1, num_col))  # 参考轨迹
root_pos = np.zeros((num_row, 3))  # 机身位置
root_rot = np.zeros((num_row, 4))  # 机身姿态
root_lin_vel = np.zeros((num_row-1, 3))  # 机身线速度
root_ang_vel = np.zeros((num_row-1, 3))  # 机身角速度
toe_pos = np.zeros((num_row, 12))  # 足端位置
dof_pos = np.zeros((num_row, 12))  # 关节位置
dof_vel = np.zeros((num_row-1, 12))  # 关节速度

go2 = utils.QuadrupedRobot()

# 默认足端初始位置（质心系）
toe_pos_init = [0.178, -0.173, -0.28, 0.178, 0.173, -0.28, -0.178, -0.173, -0.28, -0.178, 0.173, -0.28]
toe_pos[:] = toe_pos_init

# 生成足端轨迹
def get_pos_beat(num_frames, tend):
    traj = np.zeros((num_frames, 3))
    for item, t in enumerate(np.linspace(0, tend, num_frames)):
        n = np.floor((t - 0) / T)
        z = -4 * h * (t - (2 * n + 1) * T / 2) ** 2 + h
        if np.sin(2 * np.pi * t / T) > 0:
            x = -a1 * np.sin(2 * np.pi * t / T)
        else:
            x = -a2 * np.sin(2 * np.pi * t / T)
        traj[item, 0] = x
        traj[item, 2] = z
    return traj

toe_pos[:, 0:3] += get_pos_beat(num_row, num_row/fps)

# 质心轨迹
root_pos[:, 2] = 0.28
root_rot[:, 3] = 1

# 计算关节角度
q = ca.SX.sym('q', 3, 1)
for j in range(4):
    for i in range(num_row):
        pos = go2.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ go2.toe
        cost = 500 * ca.dot((toe_pos[i, 3*j:3*j+3] - pos[:3]), (toe_pos[i, 3*j:3*j+3] - pos[:3]))
        nlp = {'x': q, 'f': cost}
        S = ca.nlpsol('S', 'ipopt', nlp)
        r = S(x0=[0.1, 0.8, -1.5], lbx=go2.lb[3*j:3*j+3], ubx=go2.ub[3*j:3*j+3])
        q_opt = r['x']
        dof_pos[i, 3*j:3*j+3] = q_opt.T

# 计算关节角速度
for i in range(num_row - 1):
    dof_vel[i, :] = (dof_pos[i+1, :] - dof_pos[i, :]) * fps

# 组合轨迹
beat_ref[:, :3] = root_pos[:num_row-1, :]
beat_ref[:, 3:7] = root_rot[:num_row-1, :]
beat_ref[:, 7:10] = root_lin_vel
beat_ref[:, 10:13] = root_ang_vel
beat_ref[:, 13:25] = toe_pos[:num_row-1, :]
beat_ref[:, 25:37] = dof_pos[:num_row-1, :]
beat_ref[:, 37:49] = dof_vel



# 可视化足端 Z 坐标轨迹
times = np.linspace(0, num_row / fps, num_row)
plt.figure(figsize=(8, 5))
plt.plot(times, toe_pos[:, 2], label="Right Front Toe Z Trajectory", color='blue')
plt.xlabel("Time (s)")
plt.ylabel("Z Coordinate (m)")
plt.title("Right Front Toe Z Coordinate vs Time")
plt.legend()
plt.grid(True)
plt.show()

# 导出txt
outfile = 'output/beat_60BPM.txt'
np.savetxt(outfile, beat_ref, delimiter=',')

# 保存json文件
json_data = {
    'frame_duration': 1 / fps,
    'frames': beat_ref.tolist()
}
with open('output_json/beat_60BPM.json', 'w') as f:
    json.dump(json_data, f, indent=4)
with open('go2ST/beat_60BPM.json', 'w') as f:
    json.dump(json_data, f, indent=4)

