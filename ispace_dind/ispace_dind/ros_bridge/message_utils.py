from functools import partial
from std_msgs.msg import MultiArrayDimension, Float32MultiArray
import numpy as np
from rclpy.clock import Clock, ClockType, Time

def _numpy2multiarray(multiarray_type, np_array):
    """Convert numpy.ndarray to multiarray"""
    multiarray = multiarray_type()
    multiarray.layout.dim = [MultiArrayDimension(label="dim%d" % i, size=np_array.shape[i], stride=np_array.shape[i] * np_array.dtype.itemsize) for i in range(np_array.ndim)]
    multiarray.data = np_array.reshape(1, -1)[0].tolist()
    return multiarray

def _multiarray2numpy(pytype, dtype, multiarray):
    """Convert multiarray to numpy.ndarray"""
    dims = list(map(lambda x: x.size, multiarray.layout.dim))
    return np.array(multiarray.data, dtype=pytype).reshape(dims).astype(dtype)

numpy2f32multi = partial(_numpy2multiarray, Float32MultiArray)
f32multi2numpy = partial(_multiarray2numpy, float, np.float32)

def ros_now() -> Time:
    return Clock(clock_type=ClockType.ROS_TIME).now()

def ros_now_msg():
    return ros_now().to_msg()

def ros_now_sec() -> float:
    return ros_now().nanoseconds / 1000000000

def time2int(time: Time) -> str:
    return f'{time.sec}{str(time.nanosec//1000000).zfill(3)}'  