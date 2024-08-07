
def main():
    
    #superparameter
    
    with SharedMemoryManager() as shm_manager:
        with WSGController(shm_manager, parameters) as gripper,\
             RTDEInterpolationController(shm_manager, parameters) as controller,\
             Spacemouse(shm_manager, parameters) as sm:
            
            #ready
            state = controller.get_state()
            target_pose = state['TargetTCPPose']
            gripper_target_pos = gripper.get_state()['gripper_position']
            t_start = time.monotonic()
            gripper.restart_put(t_start-time.monotonic() + time.time())
            while True:#通过3D鼠标控制机器人
                s = time.time()
                t_cycle_end = t_start + (iter_idx + 1) * dt#dt = 1/frequency = 1/125 = 0.008
                t_sample = t_cycle_end - command_latency#command_latency = dt/2
                t_command_target = t_cycle_end + dt

                precise_wait(t_sample)
                sm_state = sm.get_motion_state_transformed()#从3D鼠标中获取位姿
                # print(sm_state)
                dpos = sm_state[:3] * (max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (max_rot_speed / frequency)
                
                drot = st.Rotation.from_euler('xyz', drot_xyz)
                target_pose[:3] += dpos
                target_pose[3:] = (drot * st.Rotation.from_rotvec(
                    target_pose[3:])).as_rotvec()

                dpos = 0
                if sm.is_button_pressed(0):
                    # close gripper
                    dpos = -gripper_speed / frequency
                if sm.is_button_pressed(1):
                    dpos = gripper_speed / frequency
                gripper_target_pos = np.clip(gripper_target_pos + dpos, 0, max_gripper_width)
                
                #执行手臂移动
                controller.schedule_waypoint(target_pose, 
                    t_command_target-time.monotonic()+time.time())
                #执行夹爪移动
                gripper.schedule_waypoint(gripper_target_pos, 
                    t_command_target-time.monotonic()+time.time())
                #等待执行完毕
                precise_wait(t_cycle_end)
                iter_idx += 1

if __name__=="__main__":
    main()

##################################################################
class RTDEInterpolationController(mp.Process):
    def __init__(self,...):
        input_queue = SharedMemoryQueue.create_from_examples()
        
        ring_buffer = SharedMemoryRingBuffer.create_from_examples()
    
    def start(self, wait=True):
        self.start_wait()
    def stop(self, wait=True): 
        message = {
            'cmd': Command.STOP.value #cmd停止机器人
        }
        self.input_queue.put(message) #输入到线程中控制
        if wait:
            self.stop_wait()
    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)#等待线程事件超时
        assert self.is_alive()
    def stop_wait(self):
        self.join()
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    def __enter__(self):
        self.start()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    # ========= command methods ============命令方法
    def servoL(self, pose, duration=0.1):#伺服
        """
        duration: desired time to reach pose期望达到姿势的时间
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    def schedule_waypoint(self, pose, target_time):#安排路径点，输入target pose
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)
    
    # ========= receive APIs =============接收APIs
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============主函数
    def run(self):
        rtde_c = RTDEControlInterface(hostname=robot_ip)#控制界面,Ur_rtde库专属
        rtde_r = RTDEReceiveInterface(hostname=robot_ip)#接收界面
        
        try:
            #设置机械臂参数
            #移动到初始位姿
            pose_interp = PoseTrajectoryInterpolator(#点跟踪插值算法
                times=[curr_t],
                poses=[curr_pose]
                )
            keep_running = True
            while keep_running:
                t_now = time.monotonic()
                pose_command = pose_interp(t_now)
                assert rtde_c.servoL(pose_command, #伺服控制移动到curr_pose
                    vel, acc, # dummy, not used by ur5
                    dt, 
                    self.lookahead_time, 
                    self.gain)
                self.ring_buffer.put(state)

                try:#共享队列获取命令
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                for i in range(n_cmd):
                    if cmd == Command.STOP.value:
                    elif cmd == Command.SERVOL.value:#伺服控制模式
                        pose_interp = pose_interp.drive_to_waypoint(#驱动到路径点
                            pose=target_pose,
                            time=t_insert,#时间差值
                            curr_time=curr_time,#当前时间
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:#路径点模式
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                    else:
                        keep_running = False
                        break
        finally:
            pass
            
            
##################################################################
class Spacemouse(mp.Process):
    def __init__(self,):
        example = {
            # 3 translation, 3 rotation, 1 period
            'motion_event': np.zeros((7,), dtype=np.int64),
            # left and right button
            'button_state': np.zeros((n_buttons,), dtype=bool),
            'receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, 
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        #共享变量
        
    def get_motion_state(self):
        state = self.ring_buffer.get()
        state = np.array(state['motion_event'][:6], 
            dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state
    
    def get_motion_state_transformed(self):
        pass
    def get_button_state(self):
        state = self.ring_buffer.get()
        return state['button_state']    
    def is_button_pressed(self, button_id):
        return self.get_button_state()[button_id]
    #========== start stop API ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()
    def __enter__(self):#实例可以通过上下文管理器来使用，确保在使用结束后能够正确地停止通信进程。
        self.start()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):#实例可以通过上下文管理器来使用，确保在使用结束后能够正确地停止通信进程。
        self.stop()
    
    def run():
        spnav_open()
        try:
            self.ring_buffer.put({
                'motion_event': motion_event,
                'button_state': button_state,
                'receive_timestamp': time.time()
            })
            self.ready_event.set()
            while not self.stop_event.is_set():
                event = spnav_poll_event()
                receive_timestamp = time.time()
                if isinstance(event, SpnavMotionEvent):
                    motion_event[:3] = event.translation
                    motion_event[3:6] = event.rotation
                    motion_event[6] = event.period
                elif isinstance(event, SpnavButtonEvent):
                    button_state[event.bnum] = event.press
                else:
                    # finish integrating this round of events
                    # before sending over
                    self.ring_buffer.put({
                        'motion_event': motion_event,
                        'button_state': button_state,
                        'receive_timestamp': receive_timestamp
                    })
                    time.sleep(1/self.frequency)
        finally:#finally块：无论try块中是否发生了异常，finally块中的代码都会被执行。
            spnav_close()