import numpy as np
import utils
import casadi as ca
import CPG
import matplotlib.pyplot as plt
import json

# 使用CPG模型设计trot步态，使用hopf振荡器获得周期性的相位信号，根据这个相位信号计算足端轨迹

num_row = 100
num_col = 49
fps = 50

trot_ref = np.ones((num_row-1, num_col))
root_pos = np.zeros((num_row, 3))
root_rot = np.zeros((num_row, 4))
root_lin_vel = np.zeros((num_row-1, 3))
root_ang_vel = np.zeros((num_row-1, 3))
root_rot_dot = np.zeros((num_row-1, 4))
toe_pos = np.zeros((num_row, 12))
dof_pos = np.zeros((num_row, 12))
dof_vel = np.zeros((num_row-1, 12))
hopf_signal = np.zeros((num_row,8))

go2 = utils.QuadrupedRobot()
initPos = np.array([-0.5, 0.5, 0.5, -0.5, 0, 0, 0, 0])
gait = 'trot'
cpg = CPG.cpgBuilder(initPos, gait=gait)

# 欧拉法获取振荡信号
t = np.linspace(0, num_row/fps, num_row)
temp = cpg.initPos.reshape(8,-1)

for i in range(num_row):
    v = cpg.hopf_osci(temp)
    temp += v/fps
    hopf_signal[i] = temp.flatten()
# 相位
phase = np.arctan2(hopf_signal[:,4:8], hopf_signal[:,0:4])

# print(phase)
# plt.figure()
# plt.plot(t, hopf_signal[:,0], linewidth=5)
# # plt.plot(t, hopf_signal[:,4], linewidth=5)
# plt.plot(t, hopf_signal[:,1], linewidth=3)
# # plt.plot(t, hopf_signal[:,5], linewidth=2)
# plt.plot(t, hopf_signal[:,2], linewidth=3)
# plt.plot(t, hopf_signal[:,3], linewidth=3)
#
# plt.show()
plt.rcParams['toolbar'] = 'none'
plt.figure()
plt.plot(t, phase[:,0], linewidth=6)
plt.plot(t, phase[:,1], linewidth=6)
plt.plot(t, phase[:,2], linewidth=2)
plt.plot(t, phase[:,3], linewidth=2)
# plt.show()

# 足端位置
vx = 0.7 #频率变慢了，前进的速度需要改一改
ax = vx*cpg.T*cpg.beta
ay = 0
az = 0.08
for i in range(4):
    for j in range(num_row):
        if phase[j,i]<0:
            p = -phase[j,i]/np.pi
            toe_pos[j, 3*i+2] = CPG.endEffectorPos_z(az,p)
        else:
            p = phase[j,i]/np.pi
            toe_pos[j,3*i] = 0

        toe_pos[j,3*i] = CPG.endEffectorPos_xy(ax,p)
        toe_pos[j,3*i+1] = CPG.endEffectorPos_xy(ay,p)

plt.figure()
plt.plot(toe_pos[:,0], toe_pos[:,2], linewidth=5)
plt.show()
# 足端相对质心的坐标
toe_pos += go2.toe_pos_init
q = ca.SX.sym('q', 3, 1)

for j in range(4):
    for i in range(num_row):
        # print(j,i)
        # toe_pos是质心系下的足端轨迹，所以欧拉角和质心都是[0, 0, 0]
        pos = go2.transrpy(q, j, [0, 0, 0], [0, 0, 0]) @ go2.toe
        cost = 500*ca.dot((toe_pos[i, 3*j:3*j+3] - pos[:3]), (toe_pos[i, 3*j:3*j+3] - pos[:3]))
        # cost = 500 * dot(([0.179183, -0.172606, 0] - pos[:3]), ([0.179183, -0.172606, 0] - pos[:3]))
        nlp = {'x': q, 'f': cost}
        S = ca.nlpsol('S', 'ipopt', nlp)
        r = S(x0 = [0.1, 0.8, -1.5], lbx = go2.lb[3*j:3*j+3], ubx = go2.ub[3*j:3*j+3])
        q_opt = r['x']
        # print(q_opt)
        # toe_pos_v = go2.transrpy(q_opt, j, [0, 0, 0], [0, 0, 0]) @ go2.toe
        # print(toe_pos_v, toe_pos[i, :3])
        dof_pos[i, 3*j:3*j+3] = q_opt.T

# 关节角速度
for i in range(num_row - 1):
    dof_vel[i,:] = (dof_pos[i+1,:] - dof_pos[i,:]) * fps

# 质心位置
x = vx*t
root_pos[:,0] = x
# root_pos[:,2] = 0.3  # 太空步
root_pos[:,2] = 0.28
# 质心速度
root_lin_vel[:,0] = vx
# 机身方向
root_rot[:,3] = 1
# 机身角速度默认为0

# 组合轨迹
trot_ref[:, :3] = root_pos[:num_row-1,:]
trot_ref[:, 3:7] = root_rot[:num_row-1,:]
trot_ref[:, 7:10] = root_lin_vel
trot_ref[:, 10:13] = root_ang_vel
trot_ref[:, 13:25] = toe_pos[:num_row-1,:]
trot_ref[:, 25:37] = dof_pos[:num_row-1,:]
trot_ref[:, 37:49] = dof_vel

# 导出txt
outfile = 'output/'+gait+'.txt'
np.savetxt(outfile, trot_ref, delimiter=',')
# a=1
# 保存json文件
json_data = {
    'frame_duration': 1 / fps,
    'frames': trot_ref.tolist()
}
with open('output_json/'+ gait +'.json', 'w') as f:
    json.dump(json_data, f, indent=4)
with open('go2ST/'+ gait +'.json', 'w') as f:
    json.dump(json_data, f, indent=4)