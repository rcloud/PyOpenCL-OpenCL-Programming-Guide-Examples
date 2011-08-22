"""
Author: Robert L Cloud, rcloud@gmail.com
http://www.robertlouiscloud.com

Created: 19/08/2011

This is a simple solver of the Laplace equation
in two dimensions using Jacobian iteration. 

We use the CPU as the primary target device in this code
"""

import pyopencl as cl
import numpy as np
import time
size = 1000

width_macro = "#define WIDTH %s" %str(size)

kernel_source = width_macro + """
__kernel void solve(__global float *u, 
                    __global float *u_new)
                    {

                    int id = get_global_id(0);

                    int y = id / WIDTH;
                    int x = id % WIDTH;

                    if (y != 0 && y != WIDTH - 1 && x != 0 && x != WIDTH - 1)
                    {
                    u_new[y * WIDTH + x] = (u[(y + 1) * WIDTH + x] +
                                           u[(y - 1) * WIDTH + x] + 
                                           u[y * WIDTH + (x + 1)] + 
                                           u[y * WIDTH + (x - 1)]) / 4;
                    }
                    }
"""

def initialize(a):
    #initialized using Chapra's values, see README
    a[0,:] = 100 #top row
    a[:,0] = 75 #left column
    a[:,a.shape[0] - 1] = 50 #right column

    
ctx = cl.create_some_context()

"""
for Jacobian iteration we need two arrays, one to store the 
values from timestep i and one for timestep values i+1
"""
u = np.zeros((size,size), dtype=np.float32)
initialize(u)
u_new=np.copy(u)

program = cl.Program(ctx, kernel_source).build()
queue = cl.CommandQueue(ctx)


mf = cl.mem_flags
#create the memory objects on the device
u_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u)
u_new_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u_new)

iteration = 0


for iteration in range(1000):
    if iteration % 2 == 0:
        program.solve(queue, (size * size,), None, u_dev, u_new_dev)
    else:
        program.solve(queue, (size * size,), None, u_new_dev, u_dev)

cl.enqueue_copy(queue, u_new, u_new_dev)
