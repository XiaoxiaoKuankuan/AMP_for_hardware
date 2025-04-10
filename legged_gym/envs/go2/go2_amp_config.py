from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import glob

# amp的原始数据
MOTION_FILES = glob.glob('datasets/mocap_motions/*')
# amp的原始数据
AMP_MOTION_FILES = glob.glob('opti_traj/output_json/*')

class GO2AMPCfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        include_history_steps = None  # Number of steps of history to include.
        num_observations = 43  # 3 + 3 + 1 +12 + 12+ 12
        num_privileged_obs = 50    # 48 # 96  # 3 + 3 + 3 + 2 + 12 + 12 +12 + 3
        # reference_state_initialization = True
        # reference_state_initialization_prob = 0.85
        # amp_motion_files = MOTION_FILES  # AMP参考数据
        motion_files = 'opti_traj/output_json'  # 我们的参考数据
        frame_duration = 1 / 50
        RSI = 1  # 参考状态初始化
        num_actions = 12
        motion_name = 'beat'

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
        # 这个意思是经过decimation个仿真周期之后控制策略才会进行一次控制
        decimation = 4


    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter



    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = 0.0
            tracking_lin_vel = 0
            tracking_ang_vel = 0
            lin_vel_z = 0.0
            ang_vel_xy = 0.0
            orientation = 0.0
            torques = -0.0002
            dof_pos_limits = -10.0


class GO2AMPCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'AMPOnPolicyRunner'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        amp_replay_buffer_size = 1000000  # 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'go2_amp_example'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'ActorCritic'
        max_iterations = 100000 # number of policy updates

        amp_reward_coef = 2.0 # AMP 奖励系数
        motion_files = 'opti_traj/output_json'  # 我们的参考数据
        amp_motion_files = motion_files
        amp_num_preload_transitions = 2000000  # AMP 预加载的轨迹转换数量（200 万个）
        amp_task_reward_lerp = 0.3  # 任务奖励与 AMP 奖励的混合系数  0.3
        amp_discr_hidden_dims = [1024, 512]  # AMP 判别器（Discriminator）隐藏层维度

        min_normalized_std = [0.05, 0.02, 0.05] * 4
        # resume_path = "logs/go2_amp_beat/Mar26_18-39-47_/model_1000.pt"


#-----------------------------beat----------------------------------------------------
class GO2AMPCfg_beat(GO2AMPCfg):
    class rewards(GO2AMPCfg.rewards):

        class scales(GO2AMPCfg.rewards.scales):
            torques = 0  #  -0.0002
            dof_pos_limits = 0  # -10.0
            tracking_lin_vel = 0 # 1.5 * 1. / (.005 * 6)
            tracking_ang_vel = 0  # 0.5 * 1. / (.005 * 6)
            termination = 0 # -1
            lin_vel_z = 0  #  -10.0  # 惩罚机器人在 Z 轴（垂直方向）上的线性速度。
            ang_vel_xy = 0  # -0.2  # 惩罚机器人在 XY 平面（水平面）上的角速度  # -0.05
            orientation = 0  # -10.  # 惩罚机器人的基座（底盘）没有保持平整
            dof_vel = -0.
            dof_acc = 0   # -2.5e-7
            collision = 0  # -1.
            action_rate = 0  #-0.01
            beat_interval_match = 300
            beat_following = 0
            calculate_joint_angle = 0  # 50
            feet_air_time = 1000
            survival = 0
            foot_lift_height = 50   # 300

    class env(GO2AMPCfg.env):
        motion_name = 'beat'

class GO2AMPCfgPPO_beat(GO2AMPCfgPPO):
    class runner(GO2AMPCfgPPO.runner):
        experiment_name = 'go2_amp_beat'
        resume_path = '/home/lw/PycharmProjects/AMP_for_robot_dance/AMP_for_hardware/logs/go2_amp_beat/Apr10_21-51-14_/model_3200.pt'


#-----------------------------trot----------------------------------------------------
class GO2AMPCfg_trot(GO2AMPCfg):
    class rewards(GO2AMPCfg.rewards):

        class scales(GO2AMPCfg.rewards.scales):
            torques = 0  #  -0.0002
            dof_pos_limits = 0  # -10.0
            tracking_lin_vel = 1.5 * 1. / (.005 * 6) # 1.5 * 1. / (.005 * 6)
            tracking_ang_vel = 0  # 0.5 * 1. / (.005 * 6)
            termination = 0 # -1
            lin_vel_z = 0  #  -10.0  # 惩罚机器人在 Z 轴（垂直方向）上的线性速度。
            ang_vel_xy = 0  # -0.2  # 惩罚机器人在 XY 平面（水平面）上的角速度  # -0.05
            orientation = 0  # -10.  # 惩罚机器人的基座（底盘）没有保持平整
            dof_vel = -0.
            dof_acc = 0   # -2.5e-7
            collision = 0  # -1.
            action_rate = 0  #-0.01


    class env(GO2AMPCfg.env):
        motion_files = 'opti_traj/output_json_trot'
        motion_name = 'trot'

class GO2AMPCfgPPO_trot(GO2AMPCfgPPO):
    class runner(GO2AMPCfgPPO.runner):
        experiment_name = 'go2_amp_trot'
        motion_files = 'opti_traj/output_json_trot'  # 我们的参考数据
        amp_motion_files = motion_files
        resume_path = '/home/lw/PycharmProjects/AMP_for_robot_dance/AMP_for_hardware/logs/go2_amp_beat/Apr09_22-29-02_/model_18400.pt'

