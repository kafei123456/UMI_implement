"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF(末端执行器) (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import av
import click
import cv2
import yaml
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.bimanual_umi_env_my import BimanualUmiEnv
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
#from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.common.pose_util import pose_to_mat, mat_to_pose
import matplotlib.pyplot as plt

OmegaConf.register_new_resolver("eval", eval, replace=True)

#解决桌子碰撞问题
def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta

#解决圆球碰撞
def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal
                
                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))

#这段代码是使用 Click 库创建一个命令行接口（Command-Line Interface, CLI）。
#这个CLI接口定义了一些命令行参数，允许用户在命令行中调用脚本时指定这些参数。
@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')#权重路径
@click.option('--output', '-o', required=True, help='Directory to save recording')#保存记录的路径
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')#机器人配置文件路径
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')#用于覆盖和调整初始条件的数据集
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')#从匹配数据集中匹配特定的场景
@click.option('--match_camera', '-mc', default=0, type=int)#匹配相机
@click.option('--camera_reorder', '-cr', default='0')#相机记录器
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")#哪个RealSense相机可视化。
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")#一开始时候是否初始化机器人关节
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")#推理的行动视界，多少步推理一次
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')#以秒为单位的每个epoch的最大持续时间
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")#控制频率(Hz)
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")#接收SapceMouse命令到在机器人上执行之间的延迟(秒)。
@click.option('-nm', '--no_mirror', is_flag=True, default=False)#没用镜子
@click.option('-sf', '--sim_fov', type=float, default=None)#仿真视场
@click.option('-ci', '--camera_intrinsics', type=str, default=None)#相机内参
@click.option('--mirror_swap', is_flag=True, default=False)#镜子交换
def main(input, output, robot_config, 
    match_dataset, match_episode, match_camera,
    camera_reorder,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency, 
    no_mirror, sim_fov, camera_intrinsics, mirror_swap):
    max_gripper_width = 0.09
    gripper_speed = 0.2
    
    # load robot config file加载机械臂配置文件
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    
    # load left-right robot relative transform#加载左右机器人相对变换（双臂时）
    tx_left_right = np.array(robot_config_data['tx_left_right'])
    tx_robot1_robot0 = tx_left_right
    
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']

    # load checkpoint#加载权重
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    print("ckpt_path = ",ckpt_path)
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter加载鱼眼转换器
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            BimanualUmiEnv(                             #虚拟环境
                output_dir=output,
                robots_config=robots_config,
                grippers_config=grippers_config,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                # latency
                camera_obs_latency=0.14943838119506836, #0.17,
                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_swap=mirror_swap,
                # action
                max_pos_speed=0.1,#0.25,#2,
                max_rot_speed=0.06,#0.16,#6,
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            # load match_dataset加载匹配数据集
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break

                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")#初始化frame

            # creating model建立模型
            # have to be done after fork to prevent 
            # duplicating CUDA context with ffmpeg nvenc使用ffmpeg功能复制CUDA上下文
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model#policy
            if cfg.training.use_ema:#使用ema？
                policy = workspace.ema_model
            policy.num_inference_steps = 16 # 16 # DDIM inference iterations推理迭代次数？
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            #print('obs_pose_rep', obs_pose_rep)
            #print('action_pose_repr', action_pose_repr)


            device = torch.device('cuda')
            policy.eval().to(device)

            print("Warming up policy inference")#准备policy推理
            obs = env.get_obs()#仿真环境获取obs
            #print("456" ,obs)
            episode_start_pose = list()
            for robot_id in range(len(robots_config)):#每个机器人初始化位姿
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(#获取umi的obs dict
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=episode_start_pose)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)#policy运行执行推理
                action = result['action_pred'][0].detach().to('cpu').numpy()#动作
                #print("888",action)
                assert action.shape[-1] == 10 * len(robots_config)
                action = get_real_umi_action(action, obs, action_pose_repr)#获取UMI的动作
                assert action.shape[-1] == 7 * len(robots_config)
                del result

            print('Ready!')
            """
            while True:
                # ========= human control loop ==========人类控制loop
                print("Human in control!")
                robot_states = env.get_robot_state()
                target_pose = np.stack([rs['TargetTCPPose'] for rs in robot_states])#虚拟环境获取每个机器人TCP位姿

                gripper_states = env.get_gripper_state()
                gripper_target_pos = np.asarray([gs['gripper_position'] for gs in gripper_states])#虚拟环境获取每个机器人手爪的位姿
                
                control_robot_idx_list = [0]#控制机器人的列表

                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing计算时间
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs机器人或控制系统通过某种方式将障碍物从特定区域中移除或清除，以便机器人可以安全地执行任务。
                    obs = env.get_obs()

                    # visualize可视化
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera{match_camera}_rgb'][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = (vis_img + match_img) / 2
                    obs_left_img = obs['camera0_rgb'][-1]
                    obs_right_img = obs['camera0_rgb'][-1]
                    vis_img = np.concatenate([obs_left_img, obs_right_img, vis_img], axis=1)
                    
                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0,0,0)
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    _ = cv2.pollKey()
                    press_events = key_counter.get_press_events()
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            # Exit human control loop
                            # hand control over to the policy
                            start_policy = True
                        elif key_stroke == KeyCode(char='e'):
                            # Next episode
                            if match_episode is not None:
                                match_episode = min(match_episode + 1, env.replay_buffer.n_episodes-1)
                        elif key_stroke == KeyCode(char='w'):
                            # Prev episode
                            if match_episode is not None:
                                match_episode = max(match_episode - 1, 0)
                        elif key_stroke == KeyCode(char='m'):
                            # move the robot
                            duration = 3.0
                            ep = match_replay_buffer.get_episode(match_episode_id)

                            for robot_idx in range(1):
                                pos = ep[f'robot{robot_idx}_eef_pos'][0]
                                rot = ep[f'robot{robot_idx}_eef_rot_axis_angle'][0]
                                grip = ep[f'robot{robot_idx}_gripper_width'][0]
                                pose = np.concatenate([pos, rot])
                                env.robots[robot_idx].servoL(pose, duration=duration)
                                env.grippers[robot_idx].schedule_waypoint(grip, target_time=time.time() + duration)
                                target_pose[robot_idx] = pose
                                gripper_target_pos[robot_idx] = grip
                            time.sleep(duration)

                        elif key_stroke == Key.backspace:
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                        elif key_stroke == KeyCode(char='a'):
                            control_robot_idx_list = list(range(target_pose.shape[0]))
                        elif key_stroke == KeyCode(char='1'):
                            control_robot_idx_list = [0]
                        elif key_stroke == KeyCode(char='2'):
                            control_robot_idx_list = [1]

                    if start_policy:#一旦打开了policy就退出人工操作的while
                        break

                    precise_wait(t_sample)
                    # get teleop command获取远程操控的命令
                    sm_state = sm.get_motion_state_transformed()
                    # print(sm_state)
                    dpos = sm_state[:3] * (0.5 / frequency)
                    drot_xyz = sm_state[3:] * (1.5 / frequency)

                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    for robot_idx in control_robot_idx_list:
                        target_pose[robot_idx, :3] += dpos
                        target_pose[robot_idx, 3:] = (drot * st.Rotation.from_rotvec(
                            target_pose[robot_idx, 3:])).as_rotvec()

                    dpos = 0
                    if sm.is_button_pressed(0):
                        # close gripper关闭手爪
                        dpos = -gripper_speed / frequency
                    if sm.is_button_pressed(1):
                        dpos = gripper_speed / frequency
                    for robot_idx in control_robot_idx_list:
                        gripper_target_pos[robot_idx] = np.clip(gripper_target_pos[robot_idx] + dpos, 0, max_gripper_width)

                    # solve collision with table解决桌子碰撞
                    for robot_idx in control_robot_idx_list:
                        solve_table_collision(
                            ee_pose=target_pose[robot_idx],
                            gripper_width=gripper_target_pos[robot_idx],
                            height_threshold=robots_config[robot_idx]['height_threshold'])
                    
                    # solve collison between two robots解决两个机器人的碰撞
                    solve_sphere_collision(
                        ee_poses=target_pose,
                        robots_config=robots_config
                    )

                    action = np.zeros((7 * target_pose.shape[0],))#目标位姿

                    for robot_idx in range(target_pose.shape[0]):#每个机器人的位姿
                        action[7 * robot_idx + 0: 7 * robot_idx + 6] = target_pose[robot_idx]
                        action[7 * robot_idx + 6] = gripper_target_pos[robot_idx]


                    # execute teleop command执行远程操控命令（通过仿真环境）
                    env.exec_actions(
                        actions=[action], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        compensate_latency=False)
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                """
                #################################################################
                #核心
                # ========== policy control loop ==============policy 控制 loop
            try:
                    # start episode开始场景
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # get current pose获取当前的位姿
                    obs = env.get_obs()#仿真环境
                    episode_start_pose = list()
                    for robot_id in range(len(robots_config)):
                        pose = np.concatenate([
                            obs[f'robot{robot_id}_eef_pos'],
                            obs[f'robot{robot_id}_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        episode_start_pose.append(pose)

                    # wait for 1/30 sec to get the closest frame actually等待1/30秒才能获得最接近的画面
                    # reduces overall latency减少总体延迟
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None
                    with open('test.txt','w') as f_txt:
                        while True:
                            # calculate timing
                            t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                            # get obs
                            obs = env.get_obs()
                            obs_timestamps = obs['timestamp']
                            #f_txt.write("Obs latenc: "+str(time.time() - obs_timestamps[-1]))
                            print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                            # run inference推理
                            with torch.no_grad():
                                s = time.time()
                                obs_dict_np = get_real_umi_obs_dict(
                                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                    obs_pose_repr=obs_pose_rep,
                                    tx_robot1_robot0=tx_robot1_robot0,
                                    episode_start_pose=episode_start_pose)
                                obs_dict = dict_apply(obs_dict_np, 
                                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                                result = policy.predict_action(obs_dict)
                                raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                                action = get_real_umi_action(raw_action, obs, action_pose_repr)
                                print('Inference latency:', time.time() - s)
                                #f_txt.write("Inference latency: "+str(time.time() - s))
                                f_txt.write("action: "+ str(action))
                                
                                show_action = False
                                if show_action:
                                    
                                    point = action[:,0:3]
                                    print("point:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>",point)
                                    x = point[:,0]
                                    y = point[:,1]
                                    z = point[:,2]
                                    fig = plt.figure()
                                    ax = fig.add_subplot(111,projection="3d")

                                    ax.scatter(x,y,z,c="r", marker="o")

                                    ax.set_xlabel("x lable")
                                    ax.set_ylabel("y lable")
                                    ax.set_zlabel("z lable")
                                    plt.show()
                                
                            
                            # convert policy action to env actions将policy操作转换为环境操作
                            this_target_poses = action
                            """
                            assert this_target_poses.shape[1] == len(robots_config) * 7
                            for target_pose in this_target_poses:
                                for robot_idx in range(len(robots_config)):
                                    solve_table_collision(
                                        ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                                        gripper_width=target_pose[robot_idx * 7 + 6],
                                        height_threshold=robots_config[robot_idx]['height_threshold']
                                    )
                                
                                # solve collison between two robots
                                solve_sphere_collision(
                                    ee_poses=target_pose.reshape([len(robots_config), -1]),
                                    robots_config=robots_config
                                )
                            """
                            # deal with timing
                            # the same step actions are always the target for保证在相同的step 动作总是一致的
                            action_timestamps = (np.arange(len(action), dtype=np.float64)
                                ) * dt + obs_timestamps[-1]
                            
                            action_exec_latency = 0.1#动作延时
                            curr_time = time.time()
                            #print("main_curr_time: ",curr_time)
                            #f_txt.write("action_timestamps:{} ".format(action_timestamps/10000.0))
                            #f_txt.write("curr_time: "+str(curr_time))

                            is_new = action_timestamps > (curr_time + action_exec_latency)
                            if np.sum(is_new) == 0:
                                # exceeded time budget, still do something超出时间预算，还是要做点什么
                                this_target_poses = this_target_poses[[-1]]
                                # schedule on next available step安排下一个可用的步骤
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamp = eval_t_start + (next_step_idx) * dt
                                print('Over budget', action_timestamp - curr_time)
                                action_timestamps = np.array([action_timestamp])
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]
                            #f_txt.write("action_timestamps_after:{} ".format(action_timestamps/10000.0))
                            #f_txt.write("this_target_poses: "+str(this_target_poses))
                            # execute actions执行动作
                            
                            # env.exec_actions(
                            #     actions=this_target_poses,
                            #     timestamps=action_timestamps,
                            #     compensate_latency=False
                            # )
                            print(f"Submitted {len(this_target_poses)} steps of actions.")

                            # visualize可视化
                            episode_id = env.replay_buffer.n_episodes
                            obs_left_img = obs['camera0_rgb'][-1]
                            obs_right_img = obs['camera0_rgb'][-1]
                            vis_img = np.concatenate([obs_left_img, obs_right_img], axis=1)
                            text = 'Episode: {}, Time: {:.1f}'.format(
                                episode_id, time.monotonic() - t_start
                            )
                            cv2.putText(
                                vis_img,
                                text,
                                (10,20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5,
                                thickness=1,
                                color=(255,255,255)
                            )
                            cv2.imshow('default', vis_img[...,::-1])

                            _ = cv2.pollKey()
                            press_events = key_counter.get_press_events()
                            stop_episode = False
                            for key_stroke in press_events:
                                if key_stroke == KeyCode(char='s'):
                                    # Stop episode停止场景
                                    # Hand control back to human手动控制
                                    print('Stopped.')
                                    stop_episode = True

                            t_since_start = time.time() - eval_t_start
                            if t_since_start > max_duration:
                                print("Max Duration reached.")#最大持续时间
                                stop_episode = True
                            if stop_episode:
                                env.end_episode()
                                break

                            # wait for execution等待执行
                            precise_wait(t_cycle_end - frame_latency)
                            iter_idx += steps_per_inference
                    f_txt.close()

            except KeyboardInterrupt:
                    print("Interrupted!")#中断执行
                    # stop robot.
                    env.end_episode()
                
            print("Stopped.")



# %%
if __name__ == '__main__':
    main()