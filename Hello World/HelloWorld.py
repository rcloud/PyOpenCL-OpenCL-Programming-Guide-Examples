"""
Author: Robert L Cloud, rcloud@gmail.com
http://www.robertlouiscloud.com

Created: 17/08/2011

A Python port of the exercise of Chapter 2  and 3 of
The OpenCL Programming Guide by Munshi et al.
"""

import pyopencl as cl
import numpy as np
import sys


kernel_source = """
    __kernel void sum(__global const float *a,
                      __global const float *b, 
                      __global float *c)
    {
      int gid = get_global_id(0);
      c[gid] = a[gid] + b[gid];
    }
"""

def python_kernel(a, b, c):
    temp = a + b
    error = False
    if np.array_equal(temp, c) == False:
        print "Error"
    else:
        print "Success!"


def no_cpu_or_gpu():
    print "You have neither a CPU nor a GPU platform registered with PyOpenCL"
    sys.exit(-1)


platforms = cl.get_platforms()
#determine CPU platform and GPU platform
cpu_platform = None
gpu_platform = None
has_cpu = False
has_gpu = False
cpu_device = None
gpu_device = None

for i in range(len(platforms)):
    if platforms[i].get_devices()[0].get_info(cl.device_info.TYPE) == cl.device_type.CPU:
        cpu_platform = platforms[i]
    elif platforms[i].get_devices()[0].get_info(cl.device_info.TYPE) == cl.device_type.GPU:
        gpu_platform = platforms[i]

if cpu_platform != None:
    has_cpu = True
if gpu_platform != None:
    has_gpu = True

if has_cpu == False and has_gpu == False:
    no_cpu_or_gpu()

#get the device of the CPU and the most powerful GPU if available

if has_cpu:
    #There will only be a single CPU device most likely
    cpu_device = cpu_platform.get_devices()[0]

if has_gpu:
    #There may be more than one GPU, let's get the most powerful(most cores)
    max_cu = 0
    index = 0
    gpus = gpu_platform.get_devices()
    for i in range(len(gpu_platform.get_devices())):
        if gpus[i].get_info(cl.device_info.MAX_COMPUTE_UNITS) > max_cu:
            max_cu = gpus[i].get_info(cl.device_info.MAX_COMPUTE_UNITS)
            index = i
    gpu_device = gpu_platform.get_devices()[index]
        

#create contexts for the CPU and GPU devices if available
#the context constructor requires a list
#command queues are limited to one device, but each device may have multiple command queues
if has_cpu:
    cpu_list = []
    cpu_list.append(cpu_device)
    cpu_context = cl.Context(devices=cpu_list)
    cpu_queue = cl.CommandQueue(cpu_context)

if has_gpu:
    gpu_list = []
    gpu_list.append(gpu_device)
    gpu_context = cl.Context(devices=gpu_list)
    gpu_queue = cl.CommandQueue(gpu_context)

size = 1000
a = np.ones(size, dtype=np.float32)
b = np.ones(size, dtype=np.float32) * 2
c = np.zeros_like(a)

#lets first run this on the CPU, so create memory objects for the cpu_context
if has_cpu:
    a_dev = cl.Buffer(cpu_context, cl.mem_flags.READ_WRITE, a.nbytes)
    b_dev = cl.Buffer(cpu_context, cl.mem_flags.READ_WRITE, b.nbytes)
    c_dev = cl.Buffer(cpu_context, cl.mem_flags.READ_WRITE, c.nbytes)

    cpu_program = cl.Program(cpu_context, kernel_source).build()
#copy memory objects to the device
    cl.enqueue_copy(cpu_queue, a_dev, a, is_blocking=True)
    cl.enqueue_copy(cpu_queue, b_dev, b, is_blocking=True)

#__call__ (queue, global_size, local_size, *args, global_offset=None, wait_for=None, g_times_|=False
    cpu_program.sum(cpu_queue, a.shape, None, a_dev, b_dev, c_dev)
    cl.enqueue_copy(cpu_queue, c, c_dev)
#check results
    python_kernel(a, b, c)

#now run on the GPU
if has_gpu:
    a_dev = cl.Buffer(gpu_context, cl.mem_flags.READ_WRITE, a.nbytes)
    b_dev = cl.Buffer(gpu_context, cl.mem_flags.READ_WRITE, b.nbytes)
    c_dev = cl.Buffer(gpu_context, cl.mem_flags.READ_WRITE, c.nbytes)

    gpu_program = cl.Program(gpu_context, kernel_source).build()
    cl.enqueue_copy(gpu_queue, a_dev, a, is_blocking=True)
    cl.enqueue_copy(gpu_queue, b_dev, b, is_blocking=True)

    gpu_program.sum(gpu_queue, a.shape, None, a_dev, b_dev, c_dev)
    cl.enqueue_copy(gpu_queue, c, c_dev)
    python_kernel(a, b, c)
