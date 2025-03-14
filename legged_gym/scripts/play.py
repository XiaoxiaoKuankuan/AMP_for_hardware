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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

from rsl_rl.modules import ActorCritic
def load_policy() -> dict:
    '''
    这个函数是按照task_list中的顺序提取对应的动作policy函数
    '''
    dance_task_policy = {}
    path_list = ['/home/pcpc/robot_dance/legged_gym/log/GO2/keep_the_beat/model_1500.pt',
                '/home/pcpc/robot_dance/legged_gym/log/GO2/pace/model_1500.pt',
                '/home/pcpc/robot_dance/legged_gym/log/GO2/swing/model_1500.pt',
                '/home/pcpc/robot_dance/legged_gym/log/GO2/trot/model_1500.pt',
                '/home/pcpc/robot_dance/legged_gym/log/GO2/turn_and_jump/model_1500.pt',
                '/home/pcpc/robot_dance/legged_gym/log/GO2/wave/model_1500.pt']
    task_list = ['go2_dance_beat', 'go2_dance_pace', 'go2_dance_swing', 'go2_dance_trot',
                 'go2_dance_turn_and_jump', 'go2_dance_wave']

    for i, load_path in enumerate(path_list):
        env_cfg = task_registry.env_cfgs[task_list[i]]
        train_cfg = task_registry.train_cfgs[task_list[i]]
        policy:torch.nn.Module = ActorCritic(env_cfg.env.num_observations, env_cfg.env.num_observations,
                                             env_cfg.env.num_actions, train_cfg.policy.actor_hidden_dims,
                                             train_cfg.policy.critic_hidden_dims, train_cfg.policy.activation,
                                             train_cfg.policy.init_noise_std).to('cuda:0')
        loaded_dict = torch.load(load_path)
        policy.load_state_dict(loaded_dict['model_state_dict'])
        policy.eval()
        dance_task_policy[task_list[i]] = policy.act_inference

    return dance_task_policy

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # env_cfg.env.debug = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    # policy_dict = load_policy()
    # policy = policy_dict[args.task]



    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        # print(actions)
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                    'foot_pos_0_x' : env.toe_pos[robot_index, 0].item(),
                    'foot_pos_0_y': env.toe_pos[robot_index, 1].item(),
                    'foot_pos_0_z': env.toe_pos[robot_index, 2].item(),
                    'foot_pos_1_x': env.toe_pos[robot_index, 3].item(),
                    'foot_pos_1_y': env.toe_pos[robot_index, 4].item(),
                    'foot_pos_1_z': env.toe_pos[robot_index, 5].item(),
                    'foot_pos_2_x': env.toe_pos[robot_index, 6].item(),
                    'foot_pos_2_y': env.toe_pos[robot_index, 7].item(),
                    'foot_pos_2_z': env.toe_pos[robot_index, 8].item(),
                    'foot_pos_3_x': env.toe_pos[robot_index, 9].item(),
                    'foot_pos_3_y': env.toe_pos[robot_index, 10].item(),
                    'foot_pos_3_z': env.toe_pos[robot_index, 11].item(),
                    'base_pos_x' : env.base_pos[robot_index, 0].item(),
                    'base_pos_y': env.base_pos[robot_index, 1].item(),
                    'base_pos_z': env.base_pos[robot_index, 2].item(),
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
