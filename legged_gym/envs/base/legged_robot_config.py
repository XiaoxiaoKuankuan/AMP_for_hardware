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

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 60
        num_privileged_obs = 132 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds
        include_history_steps = None

        debug = False
        check_contact = True
        num_leg = 4
        motion_files = 'opti_traj/output_json'
        motion_name = None
        frame_duration = 1 / 50

    class terrain:
        #  地形参数
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False  # 是否使用 课程学习（curriculum learning）
        max_curriculum = 1.  # 课程学习的最大难度级别
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]  每 10 秒 更新一次指令
        heading_command = True # if true: compute ang vel command from heading error 如果 True，那么角速度命令（ang_vel_yaw）不是直接设定的，而是从朝向误差计算得出
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}
        lowest_root_height = 0.1  # ？

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        name = "legged_robot"  # actor name ？
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True  # 随机化地面摩擦系数
        friction_range = [0.5, 1.25]  # [0.4, 2.0]

        randomize_restitution = True  # True 地面的恢复系数
        restitution_range = [0.0, 0.4]

        use_default_friction = True  # 是否使用URDF默认的关节摩擦值
        use_random_friction_value = False  # 否则 使用随机的关节摩擦值或者指定值
        joint_friction_range = [0.0, 0.02]  # 随机值范围[0.01, 1.15]
        joint_friction_value = 0.3  # 指定值
        randomize_joint_friction = False  # 是否随机化关节摩擦
        randomize_joint_friction_each_joint = False  # 是否针对每个关节单独随机化摩擦
        joint_friction_factor = [0.9, 1.1]

        use_default_damping = True  # 是否使用URDF默认的阻尼值
        use_random_damping_value = True  # 否则 使用随机的阻尼值或者指定值
        joint_damping_range = [0.1, 5.]  # 随机值范围
        joint_damping_value = 0.01  # 指定值
        randomize_joint_damping = False  # 是否随机化关节阻尼
        randomize_joint_damping_each_joint = False  # 是否针对每个关节单独随机化阻尼
        joint_damping_factor = [0.9, 1.1]

        use_default_armature = True  # 是否使用URDF默认的转子惯量值
        use_random_armature_value = True  # 否则 使用随机的转子惯量值或者指定值
        joint_armature_range = [0.01, 0.2]  # 随机值范围
        joint_armature_value = [0.0355, 0.0220, 0.1289]  # 指定值 [0.0354, 0.022, 0.0513] 新电机[0.0355, 0.0220, 0.1289]
        randomize_joint_armature = False  # 是否随机化转子惯量
        randomize_joint_armature_each_joint = False  # 是否针对每个关节单独随机化转子惯量
        joint_armature_factor = [0.9, 1.1]

        randomize_motor = True  # PD部分的随机化 等价于PD的随机化 也是所有力矩部分随机化的总开关
        motor_strength_range = [0.9, 1.1]  # [0.8, 1.2]

        randomize_torque = True  # 力矩整体的随机化
        torque_multiplier_range = [0.9, 1.1]

        randomize_motor_offset = False  # 位置偏差随机化
        motor_offset_range = [-0.035, 0.035]  # Offset to add to the motor angles

        randomize_gains = False
        stiffness_multiplier_range = [0.9, 1.1]  # Factor [0.8, 1.2] [0.2, 2] [0.5, 1.5]
        damping_multiplier_range = [0.9, 1.1]

        randomize_coulomb_friction = False
        joint_coulomb_range = [0.1, 1.0]
        joint_viscous_range = [0.1, 0.9]

        randomize_base_mass = True
        added_mass_range = [-1., 1.]

        randomize_base_com = True
        added_com_range = [-0.2, 0.2]

        randomize_link_mass = True
        added_link_mass_range = [0.0, 0.2]

        push_robots = True
        push_interval_s = 15
        push_vel = True
        max_push_vel_xy = 1.  # 0.2
        push_ang = True
        max_push_ang_vel = 0.6
        swing_roll = False  # 这个参数是针对实物上跳跃落地后抖动添加的
        max_swing_roll = 0.35  # 20°

        delay_update_global_steps = 2000
        action_delay = True  # False True
        action_curr_step = [0, 1]  # [1, 1] [1, 2]
        action_delay_view = 1  # 有viewer时(打开渲染)的delay
        action_buf_len = 8
        action_delay_range = [0, 0.75]

        RSI = True
        RSI_rand = False  # 参考轨迹基础上添加随机化
        RSI_traj_rand = True  # 初始轨迹随机化

    class rewards:
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.
            torques = -0.00001
            dof_vel = -0.
            dof_acc = -2.5e-7
            base_height = -0. 
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 2.5

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 150000 # number of policy updates

        # logging
        save_interval = 100 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt