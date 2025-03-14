import glob
import os
import json
from pybullet_utils import transformations

from legged_gym.motion_loader import motion_util
from legged_gym.motion_loader import pose3d
from legged_gym.utils.math import *
from legged_gym.utils.torch_jit_utils import *
from rsl_rl.utils import utils
simple_data = 1

class motionLoader:
    # root位置，姿态，线速度，角速度，末端相对位置，关节位置，关节速度
    POS_SIZE = 3
    ROT_SIZE = 4
    LINEAR_VEL_SIZE = 3
    ANGULAR_VEL_SIZE = 3
    TAR_TOE_POS_LOCAL_SIZE = 12 # 末端执行器相对于机身的位置 腿的位置是 右前，左前，右后，左后
    JOINT_POS_SIZE = 12
    JOINT_VEL_SIZE = 12
    # TAR_TOE_VEL_LOCAL_SIZE = 12 末端执行器的速度没用到
    ROOT_POS_START_IDX = 0
    ROOT_POS_END_IDX = ROOT_POS_START_IDX + POS_SIZE  #[0:3]

    ROOT_ROT_START_IDX = ROOT_POS_END_IDX
    ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROT_SIZE  #[3:7]

    LINEAR_VEL_START_IDX = ROOT_ROT_END_IDX
    LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE # [7:10]

    ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
    ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE # [10:13]

    TAR_TOE_POS_LOCAL_START_IDX = ANGULAR_VEL_END_IDX
    TAR_TOE_POS_LOCAL_END_IDX = TAR_TOE_POS_LOCAL_START_IDX + TAR_TOE_POS_LOCAL_SIZE # [13:25]

    JOINT_POSE_START_IDX = TAR_TOE_POS_LOCAL_END_IDX
    JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE # [25:37]

    JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
    JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE # [37:49]



    def __init__(
            self,
            device,
            time_between_frames,
            frame_duration=1/50,
            data_dir='',
            preload_transitions=False,
            num_preload_transitions=1000000,
            motion_files='datasets/motion_files2',
            ):
        """Expert dataset provides AMP observations from Dog mocap dataset.
            从参考动作中导入数据，从AMP的程序修改得到的
        time_between_frames: Amount of time in seconds between transition.
        仿真环境里的时间间隔dt
        frame_duration: 参考动作的时间间隔，1/36s，读取单文件的情况下实例化时 ！！！需要进行赋值！！！
        frame_duration设置了默认值，因为读取多文件的时候这个值我是不用的，实例化类的时候！！！不会给这个参数赋值！！！
        """
        self.device = device
        self.time_between_frames = time_between_frames
        
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_idxs = []  # 只有一条轨迹
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []
        # 读取文件信息
        # # [fl, fr, rl, rr]
        if os.path.isfile(motion_files):
            motion_data = np.loadtxt(motion_files, delimiter=',')  # frames*items (70,49)
            # 用reorder_from_pybullet_to_isaac函数把腿的顺序换一下，因为isaac gym的腿顺序是 左前，右前，左后，右后
            motion_data = self.reorder_from_pybullet_to_isaac(motion_data)  #

            # Remove first 7 observation dimensions and last 12 foot_vel (root_pos and root_orn and foot_vel). (70, 42)
            # todo: 这里还不知道取出来做什么 self.trajectories的用途不明
            self.trajectories.append(torch.tensor(
                motion_data[
                :,
                self.ROOT_ROT_END_IDX:self.JOINT_VEL_END_IDX
                ], dtype=torch.float32, device=device))
            self.trajectories_full.append(torch.tensor(  # (70, 49)
                motion_data[:, :self.JOINT_VEL_END_IDX],
                dtype=torch.float32, device=device))
            self.trajectory_idxs.append(0)
            self.trajectory_weights.append(1)
            self.trajectory_frame_durations.append(frame_duration)  # 1/36s
            traj_len = (motion_data.shape[0] - 1) * frame_duration
            self.trajectory_lens.append(traj_len)  # 约为2s
            self.trajectory_num_frames.append(float(motion_data.shape[0]))  # 70
            print(f"Loaded {traj_len}s. motion from data.")
        else:
            print(glob.glob(os.path.join(motion_files, '*')))
            for i, motion_file in enumerate(glob.glob(os.path.join(motion_files, '*'))):
                self.trajectory_names.append(motion_file.split('.')[-2])
                with open(motion_file,"r") as f:
                    motion_json = json.load(f)
                    motion_data = np.array(motion_json["frames"])
                    motion_data = self.reorder_from_pybullet_to_isaac(motion_data)
                    # 如果以后有数据的四元数不是标准的四元素，还需要把四元数归一化，把二范数变成1
                    # 由于我的四元数是标准的，我这里就没有处理
                    # Remove first 7 observation dimensions (root_pos and root_orn).
                    self.trajectories.append(torch.tensor(
                        motion_data[
                        :,
                        self.ROOT_ROT_END_IDX:self.JOINT_VEL_END_IDX
                        ], dtype=torch.float32, device=device))
                    self.trajectories_full.append(torch.tensor(
                        motion_data[:, :self.JOINT_VEL_END_IDX],
                        dtype=torch.float32, device=device))
                    self.trajectory_idxs.append(i)
                    self.trajectory_weights.append(1)
                    frame_duration = float(motion_json["frame_duration"])
                    self.trajectory_frame_durations.append(frame_duration)
                    traj_len = (motion_data.shape[0] - 1) * frame_duration
                    self.trajectory_lens.append(traj_len)
                    self.trajectory_num_frames.append(float(motion_data.shape[0]))
                print(f"Loaded {traj_len}s. motion from {motion_file}.")

        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            # 根据采样的时间，利用差值 获取当前以及下一时刻的轨迹
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)  # (len, 49)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)  # (len, 49)

            root_state = torch.cat([self.preloaded_s[:, :7], self.preloaded_s[:, 31:34], self.preloaded_s[:, 34:37]], dim=-1)
            dof_pos = self.preloaded_s[:, 7:19]
            dof_vel = self.preloaded_s[:, 37:49]
            key_body_pos = self.preloaded_s[:, 19:31].reshape(-1, 4, 3)
            self.preloaded_s = build_amp_observations(root_state, dof_pos, dof_vel, key_body_pos)

            root_state = torch.cat([self.preloaded_s_next[:, :7], self.preloaded_s_next[:, 31:34], self.preloaded_s_next[:, 34:37]], dim=-1)
            dof_pos = self.preloaded_s_next[:, 7:19]
            dof_vel = self.preloaded_s_next[:, 37:49]
            key_body_pos = self.preloaded_s_next[:, 19:31].reshape(-1, 4, 3)
            self.preloaded_s_next = build_amp_observations(root_state, dof_pos, dof_vel, key_body_pos)

            print(f'Finished preloading')


        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def reorder_from_pybullet_to_isaac(self, motion_data):
        """Convert from PyBullet ordering to Isaac ordering.

        Rearranges leg and joint order from PyBullet [FR, FL, RR, RL] to
        IsaacGym order [FL, FR, RL, RR].
        """
        root_pos = self.get_root_pos_batch(motion_data)  # (33, 3)
        root_rot = self.get_root_rot_batch(motion_data)  # (33, 4)  # (x, y, z, w)

        jp_fr, jp_fl, jp_rr, jp_rl = np.split(
            self.get_joint_pose_batch(motion_data), 4, axis=1)
        joint_pos = np.hstack([jp_fl, jp_fr, jp_rl, jp_rr])  # (33, 12)
    
        fp_fr, fp_fl, fp_rr, fp_rl = np.split(
            self.get_tar_toe_pos_local_batch(motion_data), 4, axis=1)
        foot_pos = np.hstack([fp_fl, fp_fr, fp_rl, fp_rr])  # (33, 12)

        lin_vel = self.get_linear_vel_batch(motion_data)  # (33, 3)
        ang_vel = self.get_angular_vel_batch(motion_data)  # (33, 3)

        jv_fr, jv_fl, jv_rr, jv_rl = np.split(
            self.get_joint_vel_batch(motion_data), 4, axis=1)
        joint_vel = np.hstack([jv_fl, jv_fr, jv_rl, jv_rr])  # (33, 12)
        # root位置，姿态，线速度，角速度，末端相对位置，关节位置，关节速度
        return np.hstack(
            [root_pos, root_rot,lin_vel, ang_vel, foot_pos, joint_pos, joint_vel])

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)


    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idx]
        return max(
            0, (self.trajectory_lens[traj_idx] * np.random.uniform() - subst))

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        # 这里添加一个subst应该是与后面get_full_frame_at_time_batch函数有联动
        # n = self.trajectory_num_frames[traj_idxs]这一行表示参考轨迹的列数，
        # 如果直接与时间的比例相乘，那就会导致可能出现索引值溢出的问题，因为我后面没有用这个
        # 随机采样时间的函数，所以我在后面加上了一个“-1”，表示算出来的索引值从0：end-1这个范围内
        subst = self.time_between_frames + self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]  # 采样的时间占整个轨迹的比例
        n = self.trajectories[traj_idx].shape[0]          # 轨迹的长度，70
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))  # 采样时间的索引号上下限
        frame_start = self.trajectories[traj_idx][idx_low]   # 采样时间的前后两帧的数据
        frame_end = self.trajectories[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.slerp(frame_start, frame_end, blend)   # 计算差值

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        all_frame_starts = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        all_frame_ends = torch.zeros(len(traj_idxs), self.observation_dim, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_starts[traj_mask] = trajectory[idx_low[traj_mask]]
            all_frame_ends[traj_mask] = trajectory[idx_high[traj_mask]]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        return self.slerp(all_frame_starts, all_frame_ends, blend)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectories_full[traj_idx].shape[0]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        frame_start = self.trajectories_full[traj_idx][idx_low]
        frame_end = self.trajectories_full[traj_idx][idx_high]
        blend = p * n - idx_low
        return self.blend_frame_pose(frame_start, frame_end, blend)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        p = times / self.trajectory_lens[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs]-1
        idx_low, idx_high = np.floor(p * n).astype(np.int), np.ceil(p * n).astype(np.int)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), self.POS_SIZE, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), self.POS_SIZE, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), self.ROT_SIZE, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), self.ROT_SIZE, device=self.device)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), self.JOINT_VEL_END_IDX - self.LINEAR_VEL_START_IDX, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs),  self.JOINT_VEL_END_IDX - self.LINEAR_VEL_START_IDX, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = self.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = self.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = self.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = self.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, self.LINEAR_VEL_START_IDX:self.JOINT_VEL_END_IDX]
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, self.LINEAR_VEL_START_IDX:self.JOINT_VEL_END_IDX]
        blend = torch.tensor(p * n - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)

        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = utils.quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.trajectory_idxs
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)


    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """

        root_pos0, root_pos1 = self.get_root_pos(frame0), self.get_root_pos(frame1)
        root_rot0, root_rot1 = self.get_root_rot(frame0), self.get_root_rot(frame1)
        joints0, joints1 = self.get_joint_pose(frame0), self.get_joint_pose(frame1)
        tar_toe_pos_0, tar_toe_pos_1 = self.get_tar_toe_pos_local(frame0), self.get_tar_toe_pos_local(frame1)
        linear_vel_0, linear_vel_1 = self.get_linear_vel(frame0), self.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = self.get_angular_vel(frame0), self.get_angular_vel(frame1)
        joint_vel_0, joint_vel_1 = self.get_joint_vel(frame0), self.get_joint_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = transformations.quaternion_slerp(
            root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(
            motion_util.standardize_quaternion(blend_root_rot),
            dtype=torch.float32, device=self.device)
        blend_joints = self.slerp(joints0, joints1, blend)
        blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)
        blend_joints_vel = self.slerp(joint_vel_0, joint_vel_1, blend)

        return torch.cat([
            blend_root_pos, blend_root_rot, blend_joints, blend_tar_toe_pos,
            blend_linear_vel, blend_angular_vel, blend_joints_vel])

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:  # True
                idxs = np.random.choice(
                    self.preloaded_s.shape[0], size=mini_batch_size)
                s = self.preloaded_s[idxs, :]
                # s = self.preloaded_s[idxs, self.JOINT_POSE_START_IDX:self.JOINT_VEL_END_IDX]
                # s = torch.cat([
                #     s,
                #     self.preloaded_s[idxs, self.ROOT_POS_START_IDX + 2:self.ROOT_POS_START_IDX + 3]], dim=-1)
                s_next = self.preloaded_s_next[idxs, :]
                # s_next = self.preloaded_s_next[idxs, self.JOINT_POSE_START_IDX:self.JOINT_VEL_END_IDX]
                # s_next = torch.cat([
                #     s_next,
                #     self.preloaded_s_next[idxs, self.ROOT_POS_START_IDX + 2:self.ROOT_POS_START_IDX + 3]], dim=-1)
            else:
                s, s_next = [], []
                traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
                times = self.traj_time_sample_batch(traj_idxs)
                for traj_idx, frame_time in zip(traj_idxs, times):
                    s.append(self.get_frame_at_time(traj_idx, frame_time))
                    s_next.append(
                        self.get_frame_at_time(
                            traj_idx, frame_time + self.time_between_frames))
                
                s = torch.vstack(s)
                s_next = torch.vstack(s_next)
            yield s, s_next

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self.trajectories_full[0].shape[1] - 3

    @property
    def num_motions(self):
        return len(self.trajectory_names)

    def get_root_pos(self,pose):
        return pose[self.ROOT_POS_START_IDX:self.ROOT_POS_END_IDX]

    def get_root_pos_batch(self,poses):
        return poses[:, self.ROOT_POS_START_IDX:self.ROOT_POS_END_IDX]

    def get_root_rot(self,pose):
        return pose[self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX]

    def get_root_rot_batch(self,poses):
        return poses[:, self.ROOT_ROT_START_IDX:self.ROOT_ROT_END_IDX]

    def get_joint_pose(self,pose):
        return pose[self.JOINT_POSE_START_IDX:self.JOINT_POSE_END_IDX]

    def get_joint_pose_batch(self,poses):
        return poses[:, self.JOINT_POSE_START_IDX:self.JOINT_POSE_END_IDX]

    def get_tar_toe_pos_local(self,pose):
        return pose[self.TAR_TOE_POS_LOCAL_START_IDX:self.TAR_TOE_POS_LOCAL_END_IDX]

    def get_tar_toe_pos_local_batch(self,poses):
        return poses[:, self.TAR_TOE_POS_LOCAL_START_IDX:self.TAR_TOE_POS_LOCAL_END_IDX]

    def get_linear_vel(self,pose):
        return pose[self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX]

    def get_linear_vel_batch(self,poses):
        return poses[:, self.LINEAR_VEL_START_IDX:self.LINEAR_VEL_END_IDX]

    def get_angular_vel(self,pose):
        return pose[self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX]  

    def get_angular_vel_batch(self,poses):
        return poses[:, self.ANGULAR_VEL_START_IDX:self.ANGULAR_VEL_END_IDX]  

    def get_joint_vel(self,pose):
        return pose[self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX]

    def get_joint_vel_batch(self,poses):
        return poses[:, self.JOINT_VEL_START_IDX:self.JOINT_VEL_END_IDX]  



# @torch.jit.script
def build_amp_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs=False):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    root_euler = torch.vstack(quaternion2rpy_torch(root_rot)).T

    # local_root_vel = my_quat_rotate(heading_rot, root_vel)
    # local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    local_root_vel = quat_rotate_inverse(root_rot, root_vel)  # (2048, 3)
    local_root_ang_vel = quat_rotate_inverse(root_rot, root_ang_vel)  # (2048, 3)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
                                           local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0],
                                            local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    # dof_obs = dof_to_obs(dof_pos)

    obs = torch.cat((root_h, root_rot_obs[:, -3:], local_root_vel, local_root_ang_vel, dof_pos, dof_vel, flat_local_key_pos),
                    dim=-1)
    return obs


