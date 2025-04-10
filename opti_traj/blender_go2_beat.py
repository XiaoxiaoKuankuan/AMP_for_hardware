import numpy as np
import json
import os
import sys
import utils
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# 读取 txt 文件内容
def read_file(file_path):
    # 打开文件
    with open(file_path, "r") as f:
        motion_json = json.load(f)
        motion_data = np.array(motion_json["frames"])
    return motion_data

def generate_acc_dec_trajectory(start, end, num_points):
    """
    生成一个在直线轨迹上的点序列，其中速度是先加速后减速
    :param start: 起始点（标量或数组）
    :param end: 终止点（标量或数组）
    :param num_points: 插值点数
    :return: 插值点序列
    """
    t = np.linspace(0, 1, num_points)  # 时间进度，从 0 到 1 均匀分布
    # 加速阶段 t <= 0.5，使用 s = 2 * t^2
    accel_phase = 2 * (t[t <= 0.5] ** 2)
    # 减速阶段 t > 0.5，使用 s = 1 - 2 * (1-t)^2
    decel_phase = 1 - 2 * ((1 - t[t > 0.5]) ** 2)
    # 合并加速和减速阶段
    s = np.concatenate((accel_phase, decel_phase))
    # 确保 s 维度为 (num_points, 1)，便于与 start 和 end 广播
    s = s[:, np.newaxis]
    # 生成轨迹点
    return start + (end - start) * s
# 将数据分块并赋值
def process_data(file_path):
    data = read_file(file_path)
    # 进行 reshape 操作
    reshaped_data = data.reshape((-1, 61))

    root_pos = reshaped_data[1:-1, 0:3]  # 质心位置
    root_rot = reshaped_data[1:-1, 3:7]  # 质心四元数
    root_lin_vel = reshaped_data[1:-1, 31:34]  # 质心线速度
    root_ang_vel = reshaped_data[1:-1, 34:37]  # 姿态角速度
    toe_pos = reshaped_data[1:-1, 19:31]  # 足端位置
    dof_pos = reshaped_data[1:-1, 7:19]  # 各关节角度
    # print("dof_pos is :", dof_pos)
    dof_vel = reshaped_data[1:-1, 37:49]  # 各关节角速度
    # toe_world_pos = reshaped_data[1:-1, 49:61]  # 足端位置
    # 初始化 ref 矩阵
    num_row = reshaped_data.shape[0]-2

    fps = 50
    ref = np.zeros((num_row - 1, 49))
    # # 大腿内收  不能加，足端位置就不准了
    # dof_pos[48:68, 0] = generate_acc_dec_trajectory(0, 0.3, 20)
    # dof_pos[48:68, 3] = generate_acc_dec_trajectory(0, -0.3, 20)
    # dof_pos[68:, 0] = 0.3
    # dof_pos[68:, 3] = -0.3
    # 赋值
    ref[:, :3] = root_pos[:num_row - 1, :]
    ref[:, 3:7] = root_rot[:num_row - 1, :]
    ref[:, 7:10] = root_lin_vel[:num_row - 1, :]
    ref[:, 10:13] = root_ang_vel[:num_row - 1, :]
    ref[:, 13:25] = toe_pos[:num_row - 1, :]
    ref[:, 25:37] = dof_pos[:num_row - 1, :]
    ref[:, 37:49] = dof_vel[:num_row - 1, :]

    # print('root_pos is', ref[80, :3])
    # print('root_rot is', ref[80, 3:7])
    # print('toe_pos is', ref[80, 12:25])
    # print('dof_pos is', ref[80, 25:37])
    #
    # print('dof_pos_final is', ref[258, 25:37])



    # 转换足端位置为世界系下的位置
    num_frames = root_pos.shape[0]
    num_toes = toe_pos.shape[1] // 3  # 足端数量
    toe_world_pos_trans = np.zeros((num_frames, num_toes, 3))  # 初始化足端世界坐标位置

    for i in range(num_frames):
        # 当前帧的质心四元数和位置
        quaternion = root_rot[i]  # 四元数 (w, x, y, z)
        translation = root_pos[i]  # 质心位置 (x, y, z)

        # 创建旋转矩阵
        rotation = utils.quaternion2rotm(quaternion)  # 获取四元数对应的旋转矩阵

        # 足端相对位置转换到世界坐标系
        for j in range(num_toes):
            toe_relative = toe_pos[i, j * 3:(j + 1) * 3]  # 当前足端相对质心的位置
            toe_world = np.dot(rotation, toe_relative) + translation  # 使用矩阵与向量相乘
            toe_world_pos_trans[i, j] = toe_world  # 保存结果
    num_steps = num_row


    # 绘制角速度曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_steps), root_ang_vel[:, 0], label="X-axis Angular Velocity", color='r')
    plt.plot(range(num_steps), root_ang_vel[: ,1], label="Y-axis Angular Velocity", color='g')
    plt.plot(range(num_steps), root_ang_vel[: ,2], label="Z-axis Angular Velocity", color='b')

    # 添加标签和标题
    plt.xlabel("Time Step")
    plt.ylabel("Angular Velocity (rad/s)")
    plt.title("Centroid Angular Velocity vs Time")

    # 显示图例
    plt.legend()

    # 显示图形
    # plt.show()



    # # 绘制足端轨迹
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # for j in range(num_toes):
    #     # 提取每个足端的世界系轨迹
    #     toe_trajectory = toe_world_pos_trans[:, j, :]
    #     ax.plot(toe_trajectory[:, 0], toe_trajectory[:, 1], toe_trajectory[:, 2], label=f"Toe {j + 1}")
    #
    # # 设置图形属性
    # ax.set_title("Toe Trans Positions in World Frame")
    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.set_zlabel("Z (m)")
    # ax.legend()
    # # plt.show()
    #
    #
    #
    # # 绘制足端轨迹
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # ax.plot(toe_world_pos[:, 0], toe_world_pos[:, 1], toe_world_pos[:, 2], label="Right Front Leg", color="r")
    # ax.plot(toe_world_pos[:, 3], toe_world_pos[:, 4], toe_world_pos[:, 5], label="Left Front Leg", color="g")
    # ax.plot(toe_world_pos[:, 6], toe_world_pos[:, 7], toe_world_pos[:, 8], label="Right Back Leg", color="b")
    # ax.plot(toe_world_pos[:, 9], toe_world_pos[:, 10], toe_world_pos[:, 11], label="Left Back Leg", color="y")
    #
    # # 设置图形属性
    # ax.set_title("Toe Positions in World Frame")
    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.set_zlabel("Z (m)")
    # ax.legend()
    # # plt.show()
    #
    #
    # # 创建一个3D图
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # 绘制每条腿的足端位置
    # 假设每条腿的足端位置分别存储在 toe_pos 的不同部分
    # 例如：左前腿：toe_pos[:, 0:3], 右前腿：toe_pos[:, 3:6], 左后腿：toe_pos[:, 6:9], 右后腿：toe_pos[:, 9:12]

    # ax.plot(toe_pos[:, 0], toe_pos[:, 1], toe_pos[:, 2], label="Right Front Leg", color="r")
    # ax.plot(toe_pos[:, 3], toe_pos[:, 4], toe_pos[:, 5], label="Left Front Leg", color="g")
    # ax.plot(toe_pos[:, 6], toe_pos[:, 7], toe_pos[:, 8], label="Right Back Leg", color="b")
    # ax.plot(toe_pos[:, 9], toe_pos[:, 10], toe_pos[:, 11], label="Left Back Leg", color="y")
    #
    # # 设置标签和标题
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Toe Positions of the Four Legs')
    #
    # # 显示图例
    # ax.legend()
    # 展示图形
    # plt.show()


    # 创建一个2D图
    plt.figure(figsize=(10, 6))

    # 绘制每条腿在z方向上的变化
    plt.plot(range(num_steps), toe_pos[:, 2], label="Right Front Leg (Z)", color="r", linestyle='-', marker='o',
             markersize=4, alpha=0.7)
    plt.plot(range(num_steps), toe_pos[:, 5], label="Left Front Leg (Z)", color="g", linestyle='--', marker='x',
             markersize=4, alpha=0.7)
    plt.plot(range(num_steps), toe_pos[:, 8], label="Right Back Leg (Z)", color="b", linestyle='-.', marker='s',
             markersize=4, alpha=0.7)
    plt.plot(range(num_steps), toe_pos[:, 11], label="Left Back Leg (Z)", color="y", linestyle=':', marker='^',
             markersize=4, alpha=0.7)
    # 显示时间信息（在时间点处添加标注）
    for i in range(0, num_steps, int(num_steps / 10)):  # 在10个时间步上添加标注
        plt.text(i, toe_pos[i, 2], f'{i}', fontsize=8, ha='right', va='bottom')
    # 设置标签和标题
    plt.xlabel('Time Step')
    plt.ylabel('Z Position')
    plt.title('Toe Positions in Z Direction Over Time')

    # 显示图例
    plt.legend()
    # 创建一个2D图
    plt.figure(figsize=(10, 6))
    # 绘制身体质心在z方向上的轨迹
    plt.plot(range(num_steps), root_pos[:, 2], label="Body CoM (Z)", color="k", linestyle='-', linewidth=2, alpha=0.9)
    # 设置标签和标题
    plt.xlabel('Time Step')
    plt.ylabel('Z Position')
    plt.title('Body CoM Positions in Z Direction Over Time')

    # 显示图例
    plt.legend()

    # 画出关节角速度轨迹
    # 假设我们要画的是前3个关节的角速度
    dt = 1 / fps  # 计算时间步长
    joint_labels = ['Right Front Leg 1', 'Right Front Leg 2', 'Right Front Leg 3',
                    'Left Front Leg 1', 'Left Front Leg 2', 'Left Front Leg 3',
                    'Right Rear Leg 1', 'Right Rear Leg 2', 'Right Rear Leg 3',
                    'Left Rear Leg 1', 'Left Rear Leg 2', 'Left Rear Leg 3']

    plt.figure(figsize=(14, 10))

    # 绘制每个关节的角速度
    for i in range(12):
        plt.plot(np.arange(num_row) * dt, dof_vel[:, i], label=joint_labels[i])

    plt.title('All Joint Velocity Trajectories')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    # 展示图形
    plt.show()
    return ref

# 使用示例
if __name__ == "__main__":
    file_path = "/home/lw/PycharmProjects/AMP_for_robot_dance/AMP_for_hardware/opti_traj/go2_beat_60bpm.txt"  # 替换为实际文件路径
    ref = process_data(file_path)
    fps = 50
    print("Processed ref array:")
    # 计算轨迹行数
    num_row = ref.shape[0]
    print(f"参考轨迹的行数为：{num_row}")



    # 导出txt
    outfile = 'output/beat_60BPM.txt'
    np.savetxt(outfile, ref, delimiter=',')

    # 保存json文件
    json_data = {
        'frame_duration': 1 / fps,
        'frames': ref.tolist()
    }
    with open('output_json/beat_60BPM.json', 'w') as f:
        json.dump(json_data, f, indent=4)
    with open('go2ST/beat_60BPM.json', 'w') as f:
        json.dump(json_data, f, indent=4)