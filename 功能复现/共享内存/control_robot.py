import time
import numpy as np
from multiprocessing.managers import SharedMemoryManager
import scipy.spatial.transform as st
from keyboard_share_momery import KeyboardConcroller

from umi.common.precise_sleep import precise_wait

def main():
    max_pos_speed = 0.25
    max_rot_speed = 0.6
    tcp_offset = 0
    dt = 1/frequency
    command_latency = dt / 2
    
    with SharedMemoryManager() as shm_manager:
        with KeyboardConcroller as KC:
            t_start = time.monotonic()
            iter_idx = 0
            while True:
                s = time.time()
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt
                
                precise_wait(t_sample)
                sm_state = sm.get_motion_state_transformed()
                
                dpos = sm.state[:3]*(max_pos_speed/frequency)
                drot_xyz = sm_state[3:] * (max_rot_speed / frequency)
                drot = st.Rotation.from_euler('xyz', drot_xyz)
                
                print("dpos = ",dpos)
                print("drot = ",drot)
                
                precise_wait(t_cycle_end)
                iter_idx += 1

if __name__=="__main__":
    main()

