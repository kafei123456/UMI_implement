from typing import Dict, List, Union
import numbers
from queue import (Empty, Full)
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from umi.shared_memory.shared_memory_util import ArraySpec, SharedAtomicCounter
from umi.shared_memory.shared_ndarray import SharedNDArray


class SharedMemoryQueue:
    """
    A Lock-Free FIFO Shared Memory Data Structure.一种无锁FIFO共享内存数据结构。
    Stores a sequence of dict of numpy arrays.存储一个由numpy数组组成的字典序列。
    """

    def __init__(self,
            shm_manager: SharedMemoryManager,
            array_specs: List[ArraySpec],#数据类型
            buffer_size: int
        ):

        # create atomic counter创建原子计数器
        write_counter = SharedAtomicCounter(shm_manager)
        read_counter = SharedAtomicCounter(shm_manager)
        
        # allocate shared memory分配共享内存块
        shared_arrays = dict()
        for spec in array_specs:
            key = spec.name
            assert key not in shared_arrays
            array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(buffer_size,) + tuple(spec.shape),
                dtype=spec.dtype)
            shared_arrays[key] = array
        
        self.buffer_size = buffer_size
        self.array_specs = array_specs
        self.write_counter = write_counter
        self.read_counter = read_counter
        self.shared_arrays = shared_arrays
    
    @classmethod
    def create_from_examples(cls, #类方法，用于从给定的示例数据创建一个 SharedMemoryQueue 对象。
            shm_manager: SharedMemoryManager,
            examples: Dict[str, Union[np.ndarray, numbers.Number]], 
            buffer_size: int
            ):
        specs = list()
        for key, value in examples.items():
            shape = None
            dtype = None
            if isinstance(value, np.ndarray):
                shape = value.shape
                dtype = value.dtype
                assert dtype != np.dtype('O')
            elif isinstance(value, numbers.Number):
                shape = tuple()
                dtype = np.dtype(type(value))
            else:
                raise TypeError(f'Unsupported type {type(value)}')

            spec = ArraySpec(
                name=key,
                shape=shape,
                dtype=dtype
            )
            specs.append(spec)

        obj = cls(#类实例
            shm_manager=shm_manager,
            array_specs=specs,
            buffer_size=buffer_size
            )
        return obj
    
    def qsize(self):# 返回队列中当前的数据数量。
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        n_data = write_count - read_count
        return n_data
    
    def empty(self):#返回队列是否为空。
        n_data = self.qsize()
        return n_data <= 0
    
    def clear(self):#清空队列，将读取计数器设为写入计数器的值。
        self.read_counter.store(self.write_counter.load())
    
    def put(self, data: Dict[str, Union[np.ndarray, numbers.Number]]):
        #将数据放入队列中。接收一个字典参数 data，其中键是字符串，值是 numpy 数组或数值类型。
        read_count = self.read_counter.load()
        write_count = self.write_counter.load()
        n_data = write_count - read_count
        if n_data >= self.buffer_size:
            raise Full()
        
        next_idx = write_count % self.buffer_size

        # write to shared memory
        for key, value in data.items():
            arr: np.ndarray
            arr = self.shared_arrays[key].get()
            if isinstance(value, np.ndarray):
                arr[next_idx] = value
            else:
                arr[next_idx] = np.array(value, dtype=arr.dtype)

        # update idx
        self.write_counter.add(1)
    
    def get(self, out=None) -> Dict[str, np.ndarray]:
        # 从队列中取出一条数据。可选参数 out 用于指定输出的数据对象，
        #返回一个包含取出的数据的字典。
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()

        if out is None:
            out = self._allocate_empty()

        next_idx = read_count % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            np.copyto(out[key], arr[next_idx])
        
        # update idx
        self.read_counter.add(1)
        return out

    def get_k(self, k, out=None) -> Dict[str, np.ndarray]:
        #从队列中取出指定数量 k 的数据。可选参数 out 用于指定输出的数据对象，
        #返回一个包含取出的数据的字典。
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()
        assert k <= n_data

        out = self._get_k_impl(k, read_count, out=out)
        self.read_counter.add(k)
        return out

    def get_all(self, out=None) -> Dict[str, np.ndarray]:
        #从队列中取出所有数据。可选参数 out 用于指定输出的数据对象，
        #返回一个包含取出的数据的字典。
        write_count = self.write_counter.load()
        read_count = self.read_counter.load()
        n_data = write_count - read_count
        if n_data <= 0:
            raise Empty()

        out = self._get_k_impl(n_data, read_count, out=out)
        self.read_counter.add(n_data)
        return out
    
    def _get_k_impl(self, k, read_count, out=None) -> Dict[str, np.ndarray]:
        #get_k 和 get_all 方法的实现细节，用于从队列中取出指定数量的数据。
        if out is None:
            out = self._allocate_empty(k)

        curr_idx = read_count % self.buffer_size
        for key, value in self.shared_arrays.items():
            arr = value.get()
            target = out[key]

            start = curr_idx
            end = min(start + k, self.buffer_size)
            target_start = 0
            target_end = (end - start)
            target[target_start: target_end] = arr[start:end]

            remainder = k - (end - start)
            if remainder > 0:
                # wrap around
                start = 0
                end = start + remainder
                target_start = target_end
                target_end = k
                target[target_start: target_end] = arr[start:end]

        return out
    
    def _allocate_empty(self, k=None):
        #分配一个空的数据对象，用于存储取出的数据。
        result = dict()
        for spec in self.array_specs:
            shape = spec.shape
            if k is not None:
                shape = (k,) + shape
            result[spec.name] = np.empty(
                shape=shape, dtype=spec.dtype)
        return result
