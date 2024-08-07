#目标：实现读取键盘事件控制机器人位姿移动和手爪的控制开合
import multiprocessing as mp
import numpy as np
import time
from umi.share_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
import keyboard


class KeyboardConcroller(mp.Process):
    def __init__(self, shm_manager,
                get_max_k = 30,
                frequency = 200,
                max_value = 300,
                deadzone = (0,0,0,0,0,0),
                dtype = np.float32,
                n_buttons = 2,):
            super().__init__()
            if np.issubdtype(type(deadzone), np.number):
                deadzone = np.full(6, fill_value = deadzone, dtype = dtype)
            else:
                deadzone = np.array(deadzone, dtype = dtype)
            assert (deadzone >= 0).all()
            
            self.frequency = frequency
            self.max_value = max_value
            self.dtype = dtype
            self.deadzone = deadzone
            self.n_buttons = n_buttons
            self.tx_zup_spnav = np.array([#3Dmouse空间转为机械臂基座的变换矩阵
                                    [1,0,0],
                                    [0,1,0],
                                    [0,0,1]
                                    ], dtype=dtype)
            example = {
                        'motion_event':np.zeros((7,),dtype = np.int64),
                        'button_state':np.zeros((n_buttons,), dtype = bool),
                        'receive_timestamp':time.time()
                        }
            ring_buffer = SharedMemoryRingBuffer.create_from_examples(
                        shm_manager = shm_manager,
                        example = example,
                        get_max_k = get_max_k,
                        get_time_budget = 0.2,
                        put_desired_frequency = frequency
            )
            
            self.ready_event = mp.Event()
            self.stop_event = mp.Event()
            self.ring_buffer = ring_buffer
    
    #======APIs======#
    def  get_motion_state(self):
        state = self.ring_buffer.get()
        state = np.array(state['motion_event'][:6], dtype = self.dtype)/self.max_value
        is_dead = (-self.deadzone < state)&(state < self.deadzone)
        state[is_dead] = 0
        return state
        
    def get_motion_state_transformed(self):
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]
        return tf_state
    def get_button_state(self):
        state = self.ring_buffer.get()
        return state['button_state']
    def is_button_pressed(self,button_id):
        return self.get_button_state()[button_id]
    
    #==========Process start stop API ===========    
    def start(self, wait = True):
        super().start
        if wait:
            self.ready_event.wait()
    def stop(self, wait = True):
        self.stop_event.set()
        if wait:
            self.join()
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self,exc_type,exc_val,exc_tb):
        self.stop()
        
    def run(self):
        
        try:
            motion_event = np.zeros((7,),dtype = np.int64)
            button_state = np.zeros((self.n_buttons,),dtype = bool)
            
            self.ring_buffer.put({
                'motion_event':motion_event,
                'button_state':button_state,
                'receive_timestamp':time.time()            
            })
            #开启线程
            self.ready_event.set()
            
            while not self.stop_event.is_set():
                event = keyboard.read_event()
                receive_timestamp = time.time()
                if event.event_type == keyboard.KEY_DOWN:
                    #transform
                    if event.name == 'u':
                        motion_event[0] += 0.01
                    elif event.name == 'j':
                        motion_event[0] -= 0.01
                    elif event.name == 'i':
                        motion_event[1] += 0.01
                    elif event.name == 'k':
                        motion_event[1] -= 0.01
                    elif event.name == 'o':
                        motion_event[2] += 0.01
                    elif event.name == 'l':
                        motion_event[2] -= 0.01
                        
                        
                    elif event.name == '4':
                        motion_event[3] += (np.PI/1800)
                    elif event.name == '1':
                        motion_event[3] -= (np.PI/1800)
                    elif event.name == '5':
                        motion_event[4] += (np.PI/1800)
                    elif event.name == '2':
                        motion_event[4] -= (np.PI/1800)
                    elif event.name == '6':
                        motion_event[5] += (np.PI/1800)
                    elif event.name == '3':
                        motion_event[5] -= (np.PI/1800)
                        
                    elif event.name == 'n':
                        button_state[0] = True
                    elif event.name == 'm':
                        button_state[0] = False
                        
                else:
                
                    self.ring_buffer.put({
                        'motion_event': motion_event,
                        'button_state': button_state,
                        'receive_timestamp': receive_timestamp
                    })
                    time.sleep(1/self.frequency)
        finally:
            keyboard.unhook_all()