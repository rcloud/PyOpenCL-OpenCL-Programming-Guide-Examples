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

size = 5
"""
OpenCL only offers single dimensional arrays as buffers but
our solver operates in two dimensional space.  We create 
a macro to get the value u[i][j] i being the row and j 
being the column in the matrix
"""

macro = "#define val(u,y,x) (u[y * %s + x])" %str(size)
kernel_source = macro + """
__kernel void solve(__global float *u, 
                    __global float *u_new)
                    {
                    int y = get_global_id(0);
                    int x = get_global_id(1);

                    int width = get_global_size(0);
                    if (y != 0 && y != width - 1 && x != 0 && x != width - 1)
                    {
                    val(u_new, y, x) = (val(u, y + 1, x) + 
                                        val(u, y - 1, x) + 
                                        val(u, y, x + 1) +
                                        val(u, y, x - 1)) / 4;
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

u_dev = cl.array.to_device(queue, u)
u_dev_new = cl.array.to_device(queue, u_new)

"""
mf = cl.mem_flags
#create the memory objects on the device
u_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u)
u_new_dev = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=u_new)
"""

#program.solve(queue, u.shape, None, u_dev, u_new_dev)

#cl.enqueue_copy(queue, u_new, u_new_dev)
#print u_new


