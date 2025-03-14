from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GO2Cfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 0.8,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 0.8,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        only_positive_rewards = False
        max_contact_force = 90

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0
            track_root_pos = 0.
            track_root_height = 0.5
            track_root_rot = 1.
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 1.


    class env(LeggedRobotCfg.env):
        # 旋转跳跃参考动作
        motion_files = 'opti_traj/output_json'
        motion_name = None
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 12
        num_observations = 42
        num_privileged_obs = 96

class GO2Cfg_PPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2_dance'
        resume_path = None

#-----------------------------swing----------------------------------------------------
class GO2DanceCfg_swing(GO2Cfg):
    class rewards( GO2Cfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales( GO2Cfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0
            track_root_pos = 0.
            track_root_height = 0.5
            track_root_rot = 1.
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 1.

            tracking_yaw = 1.
            # track_dof_pos = 5
            # track_dof_vel = 5
            # track_toe_pos = 10

    class env(GO2Cfg.env):
        # 旋转跳跃参考动作
        motion_name = 'swing'

    class noise(GO2Cfg.noise):
        class noise_scales(GO2Cfg.noise.noise_scales):
            dof_pos = 0.08


class GO2DanceCfg_swingPPO( GO2Cfg_PPO ):
    class runner( GO2Cfg_PPO.runner ):
        experiment_name = 'go2_dance_swing'

#-----------------------------beat----------------------------------------------------
class GO2DanceCfg_beat(GO2Cfg):
    class rewards(GO2Cfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(GO2Cfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0
            track_root_pos = 0
            track_root_rot = 2.
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 5.

    class env(GO2Cfg.env):
        motion_name = 'beat'

class GO2DanceCfg_beatPPO(GO2Cfg_PPO):
    class runner(GO2Cfg_PPO.runner):
        experiment_name = 'go2_dance_beat'
        # resume_path = "legged_gym/log/GO2_new/keep_the_beat/model_1500.pt"

#-----------------------------jump----------------------------------------------------
class GO2DanceCfg_turn_and_jump(GO2Cfg):
    class rewards(GO2Cfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        base_height_jump = 0.496

        class scales(GO2Cfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0

            lin_vel_z = -0
            track_root_pos = 1.
            track_root_height = 0
            track_root_rot = 1.2
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 1.5
            jump = 1

    class env(GO2Cfg.env):
        motion_name = 'turn_and_jump'



class GO2DanceCfg_turn_and_jumpPPO(GO2Cfg_PPO):
    class runner(GO2Cfg_PPO.runner):
        experiment_name = 'go2_dance_turn_and_jump'
        # resume_path = 'legged_gym/log/GO2_new/turn_and_jump/model_550.pt'




#-----------------------------wave----------------------------------------------------
class GO2DanceCfg_wave(GO2Cfg):
    class rewards(GO2Cfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(GO2Cfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0

            lin_vel_z = -0
            track_root_pos = 0.
            track_root_height = 1.
            track_root_rot = 1.2
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 0
            track_dof_vel = 0
            track_toe_pos = 8
            orientation = -0.0
            termination = -1.0

    class env(GO2Cfg.env):
        motion_name = 'wave'


class GO2DanceCfg_wavePPO(GO2Cfg_PPO):

    class runner(GO2Cfg_PPO.runner):
        experiment_name = 'go2_wave'
        # resume_path = "legged_gym/log/GO2_new/wave/model_1500.pt"

#-----------------------------pace----------------------------------------------------
class GO2DanceCfg_pace(GO2Cfg):
    class rewards(GO2Cfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(GO2Cfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0

            lin_vel_z = -0
            track_root_pos = 0.
            track_root_rot = 0
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 3
            track_dof_vel = 3
            track_toe_pos = 8
            termination = -1.0

    class env(GO2Cfg.env):
        motion_name = 'pace'




class GO2DanceCfg_pacePPO(GO2Cfg_PPO):
    class runner(GO2Cfg_PPO.runner):
        experiment_name = 'go2_pace'
        # resume_path = 'legged_gym/logs/go2_trot/Nov02_13-32-18_/model_1400.pt'
        # resume_path = "legged_gym/log/GO2_new/pace/model_1500.pt"

#-----------------------------trot----------------------------------------------------
class GO2DanceCfg_trot(GO2Cfg):
    class rewards(GO2Cfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(GO2Cfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0

            lin_vel_z = -0
            track_root_pos = 0.
            track_root_rot = 0
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 3
            track_dof_vel = 3
            track_toe_pos = 8
            termination = -1.0

    class env(GO2Cfg.env):
        motion_name = 'trot'


class GO2DanceCfg_trotPPO(GO2Cfg_PPO):
    class runner(GO2Cfg_PPO.runner):
        experiment_name = 'go2_trot'
        # resume_path = 'legged_gym/logs/go2_trot/Nov04_21-29-29_/model_1500.pt'
        # resume_path = "legged_gym/log/GO2_new/trot/model_1500.pt"

#-----------------------------stand----------------------------------------------------
class GO2DanceCfg_stand(GO2Cfg):
    class rewards(GO2Cfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

        class scales(GO2Cfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            feet_air_time = 0
            feet_contact_forces = -1
            action_rate = -2
            orientation = -1

            # 模仿奖励
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            track_root_pos = 3
            track_root_height = 0
            track_root_rot = 10
            track_lin_vel_ref = 0
            track_ang_vel_ref = 0
            track_dof_pos = 8
            track_dof_vel = 0
            track_toe_pos = 10

    class env(GO2Cfg.env):
        motion_name = 'stand'
        # 站立参考轨迹
        motion_files = 'opti_traj/go2ST'
        check_contact = False


class GO2DanceCfg_standPPO(GO2Cfg_PPO):
    class runner(GO2Cfg_PPO.runner):
        experiment_name = 'go2_stand'
