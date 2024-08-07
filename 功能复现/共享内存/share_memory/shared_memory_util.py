from typing import Tuple
from dataclasses import dataclass
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from atomics import atomicview, MemoryOrder, UINT

@dataclass
class ArraySpec:
    name: str
    shape: Tuple[int]
    dtype: np.dtype


class SharedAtomicCounter:
    def __init__(self, 
            shm_manager: SharedMemoryManager, 
            size :int=8 # 64bit int
            ):
        shm = shm_manager.SharedMemory(size=size)
        self.shm = shm
        self.size = size
        self.store(0) # initialize初始化

    @property
    def buf(self):#读取buffer
        return self.shm.buf[:self.size]
    
    #读取操作
    def load(self) -> int:
        with atomicview(buffer=self.buf, atype=UINT) as a: 
            value = a.load(order=MemoryOrder.ACQUIRE)
        return value
    
    #存储操作
    def store(self, value: int):
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.store(value, order=MemoryOrder.RELEASE)
        #该线程保证在存储操作之前的所有内存读写操作都将在存储操作之前执行，然后存储操作自身将被视为一个释放操作，
        #该释放操作将释放所有对共享数据的修改，并将这些修改推送到内存中。
    
    #添加操作
    def add(self, value: int):
        with atomicview(buffer=self.buf, atype=UINT) as a:
            a.add(value, order=MemoryOrder.ACQ_REL)
        #当一个线程执行增加操作，并使用 MemoryOrder.ACQ_REL 时，该线程保证在增加操作之前的所有内存读写操作都将在增加操作之前执行，
        #然后增加操作自身将被视为一个获取-释放操作，该操作既获取共享数据的当前值，又将增加后的值释放回共享内存中。