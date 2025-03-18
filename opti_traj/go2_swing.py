import numpy as np
from scipy.constants import g
from pybullet_utils.transformations import quaternion_slerp, quaternion_multiply, quaternion_conjugate
import utils
from casadi import *
import json

# 规划摇摆的轨迹，输出的文件为一个状态矩阵txt文件，列数为49列，因为读取文件的固定格式为49列
# root位置[0:3]，root姿态[3:7] [x,y,z,w]，线速度[7:10]，角速度[10:13]
# 足端位置（在世界系下相对于质心的位置）[13:25][go2:[FR, FL, RR, RL]]，panda7没看过是什么顺序
# 关节角度和关节[25:37]


# 机身右转30°，左转60°，右转60°，左转30°
go2 = utils.QuadrupedRobot()
num_row = 130
num_col = 49
fps = 50

swing_ref = np.ones((num_row-1, num_col))
root_pos = np.zeros((num_row, 3))
root_rot = np.zeros((num_row, 4))
root_lin_vel = np.zeros((num_row-1, 3))
root_ang_vel = np.zeros((num_row-1, 3))
root_rot_dot = np.zeros((num_row-1, 4))
toe_pos = np.zeros((num_row, 12))
dof_pos = np.zeros((num_row, 12))
dof_vel = np.zeros((num_row-1, 12))


# 质心轨迹
root_pos[:, 2] = 0.322
# 质心线速度 默认为0

# 姿态
# 计算姿态有点小问题，但是影响不大
q1 = [0, 0, 0, 1]  # [x,y,z,w]
q2 = [0, 0, np.sin(-np.pi / 12), np.cos(-np.pi / 12)]
q3 = [0, 0, np.sin(np.pi / 12), np.cos(np.pi / 12)]
root_rot[:] = q1
interval1 = 30
interval2 = 40
start = 0
end = start + interval2

for i in range(start, end):
    frac = (i - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q1, q2, frac)

start = end
end = start + interval1
for i in range(start, end):
    frac = (i - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q2, q3, frac)

start = end
end = start + interval1
for i in range(start, end):
    frac = (i - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q3, q2, frac)

start = end
end = start + interval1
for i in range(start, end):
    frac = (i - start) / (end - start)
    root_rot[i, :] = quaternion_slerp(q2, q3, frac)

interval3 = end  #10是朝前站立的时间0.2s


# 四元数的导数
for i in range(num_row-1):
    root_rot_dot[i,:] = (root_rot[i+1,:] - root_rot[i,:]) * fps
# 质心角速度
for i in range(num_row-1):
    root_ang_vel[i, :] = 2 * utils.quat2angvel_map(root_rot[i,:])@ root_rot_dot[i,:]



# 计算质心系下的足端轨迹
for i in range(toe_pos.shape[0]):
    toe_pos[i, :3] = np.transpose(utils.quaternion2rotm(root_rot[i,:])) @ go2.toe_pos_init[:3]
    toe_pos[i, 3:6] = np.transpose(utils.quaternion2rotm(root_rot[i,:])) @ go2.toe_pos_init[3:6]
    toe_pos[i, 6:9] = np.transpose(utils.quaternion2rotm(root_rot[i,:])) @ go2.toe_pos_init[6:9]
    toe_pos[i, 9:12] = np.transpose(utils.quaternion2rotm(root_rot[i,:])) @ go2.toe_pos_init[9:12]

# 计算关节角度
q = SX.sym('q', 3, 1)

for j in range(4):
    for i in range(num_row):
        # print(j,i)
        # toe_pos是质心系下的足端轨迹，所以欧拉角和质心都是[0, 0, 0]
        pos = go2.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ go2.toe
        cost = 500*dot((toe_pos[i, 3*j:3*j+3] - pos[:3]), (toe_pos[i, 3*j:3*j+3] - pos[:3]))
        # cost = 500 * dot(([0.179183, -0.172606, 0] - pos[:3]), ([0.179183, -0.172606, 0] - pos[:3]))
        nlp = {'x': q, 'f': cost}
        S = nlpsol('S', 'ipopt', nlp)
        r = S(x0 = [0.1, 0.8, -1.5], lbx = go2.lb[3*j:3*j+3], ubx = go2.ub[3*j:3*j+3])
        q_opt = r['x']
        # print(q_opt)
        # toe_pos_v = go2.transrpy(q_opt, j, [0, 0, 0], [0, 0, 0]) @ go2.toe
        # print(toe_pos_v, toe_pos[i, :3])
        dof_pos[i, 3*j:3*j+3] = q_opt.T

# 关节角速度
for i in range(num_row - 1):
    dof_vel[i,:] = (dof_pos[i+1,:] - dof_pos[i,:]) * fps


# 组合轨迹
swing_ref[:, :3] = root_pos[:num_row-1,:]
swing_ref[:, 3:7] = root_rot[:num_row-1,:]
swing_ref[:, 7:10] = root_lin_vel
swing_ref[:, 10:13] = root_ang_vel
swing_ref[:, 13:25] = toe_pos[:num_row-1,:]
swing_ref[:, 25:37] = dof_pos[:num_row-1,:]
swing_ref[:, 37:49] = dof_vel

# 导出txt
outfile = 'output/swing.txt'
np.savetxt(outfile, swing_ref, delimiter=',')

# 保存json文件
json_data = {
    'frame_duration': 1 / fps,
    'frames': swing_ref.tolist()
}
with open('output_json/swing.json', 'w') as f:
    json.dump(json_data, f, indent=4)
with open('go2ST/swing.json', 'w') as f:
    json.dump(json_data, f, indent=4)
