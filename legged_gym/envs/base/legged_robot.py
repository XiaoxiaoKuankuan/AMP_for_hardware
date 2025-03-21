# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg
from legged_gym.motion_loader.motion_loader import motionLoader
from rsl_rl.datasets.motion_loader import AMPLoader  # AMP 用

def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    # stack r, p, w in dim1
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz
def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:, 0]
    y = quat_angle[:, 1]
    z = quat_angle[:, 2]
    w = quat_angle[:, 3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

        # 重新加载动作数据
        self.motion_loader = motionLoader(motion_files=self.cfg.env.motion_files, device=self.device,
                                          time_between_frames=self.dt,
                                          frame_duration=self.cfg.env.frame_duration)
        self.action_id = [id for id, name in enumerate(self.motion_loader.trajectory_names) if
                          self.cfg.env.motion_name in name]
        if len(self.action_id) > 1:
            raise ValueError("select trajs more than 1")

        # self.motion_loader = AMPLoader(motion_files=self.cfg.env.amp_motion_files, device=self.device,
        #                                time_between_frames=self.dt)  # 先用AMP数据测试代码能不能用

        self.max_episode_length_s = self.motion_loader.trajectory_lens[self.action_id[0]]  # 轨迹秒
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)  # 轨迹步数

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(
                torch.arange(self.num_envs, device=self.device),
                self.obs_buf[torch.arange(self.num_envs, device=self.device)])
        obs, privileged_obs, _, _, _, _, _ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # 一阶滤波延迟
        if self.cfg.domain_rand.action_delay:
            delay = self.delay
        else:
            delay = torch.zeros(self.num_envs, 1, device=self.device)
        actions = (1 - delay) * actions + delay * self.actions

        clip_actions = self.cfg.normalization.clip_actions / self.cfg.control.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        # self.post_physics_step()
        reset_env_ids, terminal_amp_states = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(reset_env_ids, self.obs_buf[reset_env_ids])
            self.obs_buf_history.insert(self.obs_buf)
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf

        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        # return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

        return policy_obs, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        time = self.episode_length_buf.cpu().numpy() / self.max_episode_length * self.max_episode_length_s # 时间 s
        # traj_idxs = self.motion_loader.weighted_traj_idx_sample_batch(self.num_envs)
        traj_idxs = np.random.choice(self.action_id, size=self.num_envs, replace=True)  # action_id就一个 不随机
        self.frames = self.motion_loader.get_full_frame_at_time_batch(traj_idxs, time)  #得到对应帧数据

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])  # 机身系下
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.base_roll, self.base_pitch, self.base_yaw = euler_from_quaternion(self.base_quat)
        self.toe_pos_world = self.rb_states[:, self.feet_indices, 0:3].view(self.num_envs, -1)
        self.toe_pos_body[:, :3] = quat_rotate_inverse(self.base_quat, self.toe_pos_world[:, :3] - self.base_pos)
        self.toe_pos_body[:, 3:6] = quat_rotate_inverse(self.base_quat, self.toe_pos_world[:, 3:6] - self.base_pos)
        self.toe_pos_body[:, 6:9] = quat_rotate_inverse(self.base_quat, self.toe_pos_world[:, 6:9] - self.base_pos)
        self.toe_pos_body[:, 9:12] = quat_rotate_inverse(self.base_quat, self.toe_pos_world[:, 9:12] - self.base_pos)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]  # AMP
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, terminal_amp_states

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        if self.cfg.env.check_contact:
            self.reset_buf = torch.any(
                torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        if not self.cfg.domain_rand.RSI:
            self._reset_root_states(env_ids)
            self._reset_dofs(env_ids)
        else:
            if self.cfg.domain_rand.RSI_traj_rand:
                frames = self.motion_loader.get_full_frame_batch(len(env_ids))  # 随机
            else:
                time = np.zeros(len(env_ids), )
                traj_idxs = np.random.choice(self.action_id, size=len(env_ids), replace=True)  # 不随机
                frames = self.motion_loader.get_full_frame_at_time_batch(traj_idxs, time)
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)

        self._resample_commands(env_ids)

        # Randomize joint parameters
        self.randomize_motor_props(env_ids)
        self.randomize_dof_props(env_ids)
        if self.cfg.domain_rand.randomize_joint_armature | self.cfg.domain_rand.randomize_joint_friction \
                | self.cfg.domain_rand.randomize_joint_damping:
            self._refresh_actor_dof_props(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_dof_pos[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.action_history_buf[env_ids, :, :] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
        包含amp的函数在这里就是初始化的时候把机器人的位置和关节角度设置成参考动作的状态
        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = self.motion_loader.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = self.motion_loader.get_joint_vel_batch(frames)

        # self.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(frames)
        # self.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frames)  # 测试AMP用

        if self.cfg.domain_rand.RSI_rand:
            self.dof_pos[env_ids] += torch_rand_float(-0.05, 0.05, (len(env_ids), self.num_dof), device=self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        root_pos = self.motion_loader.get_root_pos_batch(frames)
        # root_pos = AMPLoader.get_root_pos_batch(frames)  # 测试AMP用
        root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]  # 加上每个环境的原点位置偏移量
        # 记录初始位置
        self.origin_xy[env_ids, :] = root_pos
        self.root_states[env_ids, :3] = root_pos
        # if self.cfg.domain_rand.RSI_rand:
        #     self.root_states[env_ids, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        root_orn = self.motion_loader.get_root_rot_batch(frames)

        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = quat_rotate(root_orn,self.motion_loader.get_linear_vel_batch(frames)) # 世界系下
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, self.motion_loader.get_angular_vel_batch(frames))

        # root_orn = AMPLoader.get_root_rot_batch(frames) # 测试AMP用
        # self.root_states[env_ids, 7:10] = quat_rotate(root_orn, AMPLoader.get_linear_vel_batch(frames))
        # self.root_states[env_ids, 10:13] = quat_rotate(root_orn, AMPLoader.get_angular_vel_batch(frames)) # 测试AMP用

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def get_amp_observations(self):
        # 机身位置  机身姿态 机身线速度 机身角速度 足端相对位置 关节位置 关节角速度
        # base_pos = self.base_pos - self.env_origins  # 世界系
        # base_euler = self.base_quat  # 世界系
        base_lin_vel = self.root_states[:, 7:10]
        base_lin_ang = self.root_states[:, 10:13]
        foot_pos = self.toe_pos_body
        leg_dof_pos = self.dof_pos[:, 0:12]  # LF RF LH RH
        leg_dof_vel = self.dof_vel[:, 0:12]

        return torch.cat((base_lin_vel, base_lin_ang, foot_pos, leg_dof_pos, leg_dof_vel), dim=-1)

    def compute_observations(self):
        """ Computes observations
        """
        # base: pos quat lin_vel ang_vel
        # base_pos_error = self.base_pos - self.env_origins - self.frames[:, 0:3]
        # base_euler_error = get_euler_xyz_tensor(self.base_quat) - get_euler_xyz_tensor(self.frames[:, 3:7])
        # base_lin_vel_error = self.base_lin_vel - quat_rotate_inverse(self.frames[:, 3:7], self.frames[:, 7:10])
        # base_lin_ang_error = self.base_ang_vel - quat_rotate_inverse(self.frames[:, 3:7], self.frames[:, 10:13])
        # # foot: pos q dq
        # foot_pos_error = self.toe_pos_body - self.frames[:, 13:25]
        # leg_dof_pos_error = self.dof_pos[:, 0:12] - self.frames[:, 25:37]  # LF RF LH RH
        # leg_dof_vel_error = self.dof_vel[:, 0:12] - self.frames[:, 37:49]
        #
        # tracking_error = torch.cat((base_pos_error, base_euler_error, base_lin_vel_error, base_lin_ang_error,
        #                             foot_pos_error, leg_dof_pos_error, leg_dof_vel_error), dim=-1)

        self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                                             self.base_ang_vel * self.obs_scales.ang_vel,  # 3
                                             self.projected_gravity,  # 3
                                             # self.commands[:, :3] * self.commands_scale,
                                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                             # 12
                                             self.dof_vel * self.obs_scales.dof_vel,  # 12
                                             self.actions,  # 12
                                             # self.action_history_buf[:,-1],
                                             self.base_euler_xyz * self.obs_scales.quat,  # 3
                                             # tracking_error  # 48
                                             ), dim=-1)
        self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,  # 3   # 3
                                  self.projected_gravity,  # 3   # 6
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # 12   # 18
                                  self.dof_vel * self.obs_scales.dof_vel,  # 12  # 30
                                  self.actions  # 12  # 42
                                  # self.action_history_buf[:,-1]
                                  ), dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        # print(self.obs_buf) # use in debug

    # def compute_observations(self):  # 测试AMP临时用
    #     """ Computes observations
    #     """
    #     self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
    #                                          self.base_ang_vel * self.obs_scales.ang_vel,
    #                                          self.projected_gravity,
    #                                          self.commands[:, :3] * self.commands_scale,
    #                                          (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
    #                                          self.dof_vel * self.obs_scales.dof_vel,
    #                                          self.actions
    #                                          ), dim=-1)
    #     # add perceptive inputs if not blind
    #     if self.cfg.terrain.measure_heights:
    #         heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
    #                              1.) * self.obs_scales.height_measurements
    #         self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
    #
    #     # add noise if needed
    #     if self.add_noise:
    #         self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec
    #
    #     # Remove velocity observations from policy observation.
    #     if self.num_obs == self.num_privileged_obs - 6:
    #         self.obs_buf = self.privileged_obs_buf[:, 6:]
    #     else:
    #         self.obs_buf = torch.clone(self.privileged_obs_buf)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def randomize_motor_props(self, env_ids):
        if self.cfg.domain_rand.randomize_motor:
            # Randomise the motor strength:
            if self.cfg.domain_rand.randomize_torque:
                torque_multiplier_range = self.cfg.domain_rand.torque_multiplier_range
                self.torque_multi[env_ids] = torch_rand_float(torque_multiplier_range[0], torque_multiplier_range[1],
                                                              (len(env_ids), self.num_actions), device=self.device)

            if self.cfg.domain_rand.randomize_motor_offset:
                min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
                self.motor_offsets[env_ids, :] = torch_rand_float(min_offset, max_offset,
                                                                  (len(env_ids), self.num_actions), device=self.device)

            if self.cfg.domain_rand.randomize_gains:
                p_gains_factor = self.cfg.domain_rand.stiffness_multiplier_range
                self.p_gains_all[env_ids] = torch_rand_float(p_gains_factor[0], p_gains_factor[1],
                                                             (len(env_ids), self.num_actions), device=self.device) * \
                                            self.d_gains_all[env_ids]
                d_gains_factor = self.cfg.domain_rand.damping_multiplier_range
                self.d_gains_all[env_ids] = torch_rand_float(d_gains_factor[0], d_gains_factor[1],
                                                             (len(env_ids), self.num_actions), device=self.device) * \
                                            self.d_gains_all[env_ids]

            if self.cfg.domain_rand.randomize_coulomb_friction:
                joint_coulomb_range = self.cfg.domain_rand.joint_coulomb_range
                self.joint_coulomb[env_ids] = torch_rand_float(joint_coulomb_range[0], joint_coulomb_range[1],
                                                               (len(env_ids), self.num_actions), device=self.device)
                joint_viscous_range = self.cfg.domain_rand.joint_viscous_range
                self.joint_viscous[env_ids] = torch_rand_float(joint_viscous_range[0], joint_viscous_range[1],
                                                               (len(env_ids), self.num_actions), device=self.device)

    def randomize_dof_props(self, env_ids):
        # 生成随机的关节属性因子，如摩擦力、阻尼和转动惯量
        if self.cfg.domain_rand.randomize_joint_friction:
            if self.cfg.domain_rand.randomize_joint_friction_each_joint:
                for i in range(self.num_dof):
                    factor_key = f'joint_{i + 1}_friction_factor'
                    joint_friction_factor = getattr(self.cfg.domain_rand, factor_key)
                    self.joint_friction_factor[env_ids, i] = torch_rand_float(joint_friction_factor[0],
                                                                              joint_friction_factor[1],
                                                                              (len(env_ids), 1),
                                                                              device=self.device).reshape(-1)
            else:
                joint_friction_factor = self.cfg.domain_rand.joint_friction_factor
                self.joint_friction_factor[env_ids] = torch_rand_float(joint_friction_factor[0],
                                                                       joint_friction_factor[1],
                                                                       (len(env_ids), 1), device=self.device)

        if self.cfg.domain_rand.randomize_joint_damping:
            if self.cfg.domain_rand.randomize_joint_damping_each_joint:
                for i in range(self.num_dof):
                    factor_key = f'joint_{i + 1}_damping_factor'
                    joint_damping_factor = getattr(self.cfg.domain_rand, factor_key)
                    self.joint_damping_factor[env_ids, i] = torch_rand_float(joint_damping_factor[0],
                                                                             joint_damping_factor[1],
                                                                             (len(env_ids), 1),
                                                                             device=self.device).reshape(-1)
            else:
                joint_damping_factor = self.cfg.domain_rand.joint_damping_factor
                self.joint_damping_factor[env_ids] = torch_rand_float(joint_damping_factor[0],
                                                                      joint_damping_factor[1],
                                                                      (len(env_ids), 1), device=self.device)

        if self.cfg.domain_rand.randomize_joint_armature:
            if self.cfg.domain_rand.randomize_joint_armature_each_joint:
                for i in range(self.num_dof):
                    factor_key = f'joint_{i + 1}_armature_factor'
                    joint_armature_factor = getattr(self.cfg.domain_rand, factor_key)
                    self.joint_armature_factor[env_ids, i] = torch_rand_float(joint_armature_factor[0],
                                                                              joint_armature_factor[1],
                                                                              (len(env_ids), 1),
                                                                              device=self.device).reshape(-1)
            else:
                joint_armature_factor = self.cfg.domain_rand.joint_armature_factor
                self.joint_armature_factor[env_ids] = torch_rand_float(joint_armature_factor[0],
                                                                       joint_armature_factor[1],
                                                                       (len(env_ids), 1), device=self.device)

    def _refresh_actor_dof_props(self, env_ids):
        # 应用随机生成的因子，并将它们更新到机器人的关节物理属性上
        # 遍历所有环境ID
        for env_id in env_ids:
            # 获取该环境中机器人模型的DOF属性（Degree of Freedom，关节属性）
            dof_props = self.gym.get_actor_dof_properties(self.envs[env_id], 0)

            # 遍历每个关节的DOF属性
            for i in range(self.num_dof):
                # 如果需要随机化关节摩擦力
                if self.cfg.domain_rand.randomize_joint_friction:
                    if self.cfg.domain_rand.randomize_joint_friction_each_joint:
                        # 对每个关节进行不同的摩擦力随机化
                        dof_props["friction"][i] = self.joint_friction[env_id, i] * self.joint_friction_factor[
                            env_id, i]
                    else:
                        # 所有关节使用相同的摩擦力随机化因子
                        dof_props["friction"][i] = self.joint_friction[env_id, i] * self.joint_friction_factor[
                            env_id, 0]

                # 如果需要随机化关节阻尼
                if self.cfg.domain_rand.randomize_joint_damping:
                    if self.cfg.domain_rand.randomize_joint_damping_each_joint:
                        # 对每个关节进行不同的阻尼随机化
                        dof_props["damping"][i] = self.joint_damping[env_id, i] * self.joint_damping_factor[env_id, i]
                    else:
                        # 所有关节使用相同的阻尼随机化因子
                        dof_props["damping"][i] = self.joint_damping[env_id, i] * self.joint_damping_factor[env_id, 0]

                # 如果需要随机化关节转动惯量（armature）
                if self.cfg.domain_rand.randomize_joint_armature:
                    if self.cfg.domain_rand.randomize_joint_armature_each_joint:
                        # 对每个关节进行不同的转动惯量随机化
                        dof_props["armature"][i] = self.joint_armature[env_id, i] * self.joint_armature_factor[
                            env_id, i]
                    else:
                        # 所有关节使用相同的转动惯量随机化因子
                        dof_props["armature"][i] = self.joint_armature[env_id, i] * self.joint_armature_factor[
                            env_id, 0]

            # 将更新后的DOF属性应用到该环境中的机器人
            self.gym.set_actor_dof_properties(self.envs[env_id], self.actor_handles[env_id], dof_props)

    def _process_rigid_shape_props(self, props, env_id):
        # 随机化 刚体形状 的属性，如摩擦系数（friction）和恢复系数（restitution）
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1),
                                                    device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                num_buckets = 64  # 64 256
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets,1), device='cpu')
                self.restitution_coeffs = restitution_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        # 随机化 自由度（DOF）属性，包括位置限制（position limits）、速度限制（velocity limits）、
        # 力矩限制（torque limits）、摩擦（friction）、阻尼（damping）和转子惯量（armature）
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                # print([props["lower"][i].item(), props["upper"][i].item()])
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit

        # 关节摩擦
        for i in range(self.num_dof):
            if self.cfg.domain_rand.use_default_friction:
                if env_id == 0:
                    print(f"Joint {i} use default friction value: {props['friction'][i]}")
            else:
                if self.cfg.domain_rand.use_random_friction_value:
                    if self.cfg.domain_rand.randomize_joint_friction_each_joint:
                        props["friction"][i] = self.joint_friction[env_id, i]
                    else:
                        props["friction"][i] = self.joint_friction[env_id, 0]
                else:
                    props["friction"][i] = self.cfg.domain_rand.joint_friction_value
                if env_id == 0:
                    print(f"Joint {i} use specified friction value: {props['friction'][i]}")
            self.joint_friction[env_id, i] = props["friction"][i].item()
        # 关节阻尼
        for i in range(self.num_dof):
            if self.cfg.domain_rand.use_default_damping:
                if env_id == 0:
                    print(f"Joint {i} use default damping value: {props['damping'][i]}")
            else:
                if self.cfg.domain_rand.use_random_damping_value:
                    if self.cfg.domain_rand.randomize_joint_damping_each_joint:
                        props["damping"][i] = self.joint_damping[env_id, i]
                    else:
                        props["damping"][i] = self.joint_damping[env_id, 0]
                else:
                    props["damping"][i] = self.cfg.domain_rand.joint_damping_value
                if env_id == 0:
                    print(f"Joint {i} use specified damping value: {props['damping'][i]}")
            self.joint_damping[env_id, i] = props["damping"][i].item()
        # 电机转子惯量
        for i in range(self.num_dof):
            if self.cfg.domain_rand.use_default_armature:
                if env_id == 0:
                    print(f"Joint {i} use default armature value: {props['armature'][i]}")
            else:
                if self.cfg.domain_rand.use_random_armature_value:
                      if self.cfg.domain_rand.randomize_joint_armature_each_joint:
                          props["armature"][i] = self.joint_armature[env_id, i]
                      else:
                          props["armature"][i] = self.joint_armature[env_id, 0]
                else:
                      if i==0:
                          joint_armature_value = self.cfg.domain_rand.joint_armature_value
                          for s in range(self.cfg.env.num_leg):
                              props["armature"][0+s*3] = joint_armature_value[0]
                              props["armature"][1+s*3] = joint_armature_value[1]
                              props["armature"][2+s*3] = joint_armature_value[2]
                if env_id == 0:
                    print(f"Joint {i} use specified armature value: {props['armature'][i]}")
            self.joint_armature[env_id, i] = props["armature"][i].item()
        return props

    def _process_rigid_body_props(self, props, env_id):
        # 随机化 刚体 的物理属性，主要包括质量（mass）、质心（center of mass）和连杆质量。
        if env_id==0:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                print(f"Mass of body {i}: {p.mass} (before randomization)")
            print(f"Total mass {sum} (before randomization)")

        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])

        # randomize base com
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)

        # randomize links mass
        if self.cfg.domain_rand.randomize_link_mass:
            for i in range(1, len(props)):
                rng_link_mass = self.cfg.domain_rand.added_link_mass_range
                links_mass = np.random.uniform(rng_link_mass[0], rng_link_mass[1], size=(1, ))
                props[i].mass += links_mass

        if env_id == self.num_envs-1:
            total_mass = 0
            for i in range(0, len(props)):
                total_mass += props[i].mass
            print("URDF total mass after randomization: ", total_mass)
        return props

    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
            随机生成新的运动指令
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero  如果速度大小 小于 0.2，则设置为 0，防止指令过小导致机器人停滞
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if control_type=="P":
            if self.cfg.domain_rand.randomize_motor:
                # torques = self.motor_strength[0] * self.p_gains_all*(actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.motor_strength[1] * self.d_gains_all*self.dof_vel
                torques = (self.motor_strength[0] * self.p_gains_all * (
                            actions_scaled + self.default_dof_pos_all - self.dof_pos + self.motor_offsets)
                           - self.motor_strength[
                               1] * self.d_gains_all * self.dof_vel - self.joint_coulomb * self.dof_vel - self.joint_viscous) * self.torque_multi
            else:
                torques = self.p_gains_all * (
                            actions_scaled + self.default_dof_pos_all - self.dof_pos) - self.d_gains_all * self.dof_vel
        elif control_type=="V":
            torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * (
                    self.dof_vel - self.last_dof_vel) / self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2),
                                                              device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6),
                                                           device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        if self.cfg.domain_rand.push_vel:
            max_vel = self.cfg.domain_rand.max_push_vel_xy
            self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        if self.cfg.domain_rand.push_ang:
            max_angular = self.cfg.domain_rand.max_push_ang_vel
            self.root_states[:, 10:13] = torch_rand_float(-max_angular, max_angular, (self.num_envs, 3), device=self.device) # ang vel
        if self.cfg.domain_rand.swing_roll:
            max_angular = self.cfg.domain_rand.max_swing_roll
            contact = self.contact_forces[:, self.feet_indices, 2] > 5.
            if torch.all(contact):
                self.root_states[:, 10] = torch_rand_float(-max_angular, max_angular, (self.num_envs, ),device=self.device).squeeze(1)  # roll ang vel
        if self.cfg.domain_rand.push_vel or self.cfg.domain_rand.push_ang:
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2],
                                           dim=1) * self.max_episode_length_s * 0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids],
                                                                      self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids],
                                                              0))  # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):  #没用
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:18] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[18:30] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[30:42] = 0.  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        self.action_scale = torch.tensor(self.cfg.control.action_scale, device=self.device)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rb_states = gymtorch.wrap_tensor(_rb_states).view(self.num_envs, -1, 13)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_actions,
                                                                     dtype=torch.float, device=self.device,
                                                                     requires_grad=False) + str_rng[0]

        self.action_history_buf = torch.zeros(self.num_envs, self.cfg.domain_rand.action_buf_len, self.num_dofs,
                                              device=self.device, dtype=torch.float)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos_all = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float,
                                               device=self.device, requires_grad=False)
        self.p_gains_all = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float,
                                       device=self.device, requires_grad=False)
        self.d_gains_all = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float,
                                       device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.default_dof_pos_all[:] = self.default_dof_pos[0]
        self.p_gains = self.p_gains.unsqueeze(0)
        self.p_gains_all[:] = self.p_gains[0]
        self.d_gains = self.d_gains.unsqueeze(0)
        self.d_gains_all[:] = self.d_gains[0]

        self.torque_multi = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                       requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.joint_coulomb = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.joint_viscous = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.randomize_motor_props(torch.arange(self.num_envs, device=self.device))

        # 定义参考动作帧
        self.frames = None
        # 定义初始位置
        self.origin_xy = torch.zeros_like(self.base_pos)
        self.toe_pos_body = torch.zeros((self.num_envs, 12), device=self.device)

        action_delay_range = self.cfg.domain_rand.action_delay_range
        self.delay = torch_rand_float(action_delay_range[0], action_delay_range[1], (self.num_envs, 1),
                                      device=self.device)

    # def foot_position_in_hip_frame(self, angles, l_hip_sign=1):
    #     theta_ab, theta_hip, theta_knee = angles[:, 0], angles[:, 1], angles[:, 2]
    #     l_up = 0.2
    #     l_low = 0.2
    #     l_hip = 0.08505 * l_hip_sign
    #     leg_distance = torch.sqrt(l_up ** 2 + l_low ** 2 +
    #                               2 * l_up * l_low * torch.cos(theta_knee))
    #     eff_swing = theta_hip + theta_knee / 2
    #
    #     off_x_hip = -leg_distance * torch.sin(eff_swing)
    #     off_z_hip = -leg_distance * torch.cos(eff_swing)
    #     off_y_hip = l_hip
    #
    #     off_x = off_x_hip
    #     off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
    #     off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip
    #     return torch.stack([off_x, off_y, off_z], dim=-1)

    # def foot_positions_in_base_frame(self, foot_angles):
    #     foot_positions = torch.zeros_like(foot_angles)
    #     for i in range(4):
    #         foot_positions[:, i * 3:i * 3 + 3].copy_(
    #             self.foot_position_in_hip_frame(foot_angles[:, i * 3: i * 3 + 3], l_hip_sign=(-1) ** (i)))
    #     foot_positions = foot_positions + HIP_OFFSETS.reshape(12, ).to(self.device)
    #     return foot_positions  #  原AMP用：计算足端在机身坐标系下的位置

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # prepare friction randomization
        if self.cfg.domain_rand.use_random_friction_value:
            joint_friction_range = self.cfg.domain_rand.joint_friction_range
            self.joint_friction = torch_rand_float(joint_friction_range[0], joint_friction_range[1],
                                                   (self.num_envs, self.num_dof), device=self.device)
        else:
            self.joint_friction = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        if self.cfg.domain_rand.randomize_joint_friction:
            if self.cfg.domain_rand.randomize_joint_friction_each_joint:
                self.joint_friction_factor = torch.ones(self.num_envs, self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False)
            else:
                self.joint_friction_factor = torch.ones(self.num_envs, 1, dtype=torch.float,
                                                        device=self.device, requires_grad=False)
        # prepare damping randomization
        if self.cfg.domain_rand.use_random_damping_value:
            joint_damping_range = self.cfg.domain_rand.joint_damping_range
            self.joint_damping = torch_rand_float(joint_damping_range[0], joint_damping_range[1],
                                                  (self.num_envs, self.num_dof), device=self.device)
        else:
            self.joint_damping = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        if self.cfg.domain_rand.randomize_joint_damping:
            if self.cfg.domain_rand.randomize_joint_damping_each_joint:
                self.joint_damping_factor = torch.ones(self.num_envs, self.num_dof, dtype=torch.float,
                                                       device=self.device, requires_grad=False)
            else:
                self.joint_damping_factor = torch.ones(self.num_envs, 1, dtype=torch.float,
                                                       device=self.device, requires_grad=False)
        # prepare armature randomization
        self.joint_armature = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        if self.cfg.domain_rand.randomize_joint_armature:
            joint_armature_range = self.cfg.domain_rand.joint_armature_range
            self.joint_armature = torch_rand_float(joint_armature_range[0], joint_armature_range[1],
                                                   (self.num_envs, self.num_dof), device=self.device)
            if self.cfg.domain_rand.randomize_joint_armature_each_joint:
                self.joint_armature_factor = torch.ones(self.num_envs, self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False)
            else:
                self.joint_armature_factor = torch.ones(self.num_envs, 1, dtype=torch.float,
                                                        device=self.device, requires_grad=False)
        self.randomize_dof_props(torch.arange(self.num_envs, device=self.device))


        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0],
                                                                         feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                      self.actor_handles[0],
                                                                                      penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.actor_handles[0],
                                                                                        termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_track_root_pos(self):
        # 奖励跟踪root的位置，self.base_pos装的也是绝对坐标
        # print(self.frames[:, 0:3])
        # print(self.base_pos - self.env_origins)
        return torch.exp(-20 * torch.sum(torch.square(self.frames[:, 0:3] - (self.base_pos - self.env_origins)), dim=1))

    def _reward_track_root_height(self):
        # 奖励跟踪root的高度，self.base_pos装的也是绝对坐标
        return torch.exp(-20 * torch.square(self.frames[:, 2] - self.base_pos[:, 2]))

    def _reward_track_root_rot(self):
        # 奖励跟踪root方向
        base_euler_error = get_euler_xyz_tensor(self.base_quat) - get_euler_xyz_tensor(self.frames[:, 3:7])
        rew = torch.exp(-50 * torch.sum(torch.square(base_euler_error), dim=1))
        # print(base_euler_error)
        # print(rew)
        return rew

    def _reward_track_toe_pos(self):
        # 跟踪末端执行器的相对位置
        # rb_states里面装的是绝对坐标
        # 使用quat_rotate_inverse将世界系下的末端相对足端位置转换为body系下的相对位置
        # rb_states里的数据滞后于base_pos,还没弄清楚：post_physics_step中一进去就会更新函数()，保证数据最新
        temp = torch.exp(-50 * torch.sum(torch.square(self.frames[:, 13:25] - self.toe_pos_body), dim=1))
        # print(f'ref toe pos {self.frames[:, 13:25]}')
        # print(f'toe pos {self.toe_pos_body}')
        # print(50*'*')
        return temp

    def _reward_track_dof_pos(self):
        return torch.exp(-5 * torch.sum(torch.square(self.frames[:, 25:37] - self.dof_pos[:, :12]), dim=1))

    def _reward_tracking_yaw(self):
        _, _, yaw_ref = euler_from_quaternion(self.frames[:, 3:7])
        rew = torch.exp(-torch.abs(yaw_ref - self.base_yaw))
        return rew