from __future__ import print_function
import numpy as np
import time
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Set the colormap
# plt.rcParams['image.cmap'] = 'BrBG'
plt.rcParams['image.cmap'] = 'jet'  # you can try: colourMap = plt.cm.coolwarm
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Set colour interpolation and colour map
colorinterpolation = 100
colourMap = plt.cm.jet  # you can try: colourMap = plt.cm.coolwarm

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule

# Basic parameters
a = 0.1  # Diffusion constant

timesteps = 10000  # Number of time-steps to evolve system
image_interval = 1000  # Write frequency for png files

# Grid spacings
dx = 0.01
dy = 0.01
dx2 = dx ** 2
dy2 = dy ** 2

# For stability, this is the largest interval possible
# for the size of the time-step:
dt = dx2 * dy2 / (2 * a * (dx2 + dy2))

# Boundary condition
Ttop = 100
Tbottom = 0
Tleft = 0
Tright = 30

# Initial guess of interior grid
Tguess = 0.25

# Set Dimension
lenX = lenY = 400  # we set it rectangular
# Set meshgrid
X, Y = np.meshgrid(np.arange(0, lenX), np.arange(0, lenY))

def write_field(field, step):
    plt.gca().clear()
    # plt.clf()

    plt.figure(dpi=300)
    # Configure the contour
    plt.title("Contour of Temperature")
    plt.contourf(X, Y, field, colorinterpolation, cmap=colourMap)
    # Set Colorbar
    plt.colorbar()
    plt.axis('on')
    plt.savefig('heat_PyCUDA_{0:03d}.png'.format(step))

a = np.float32(a)
lenX = np.int32(lenX)
lenY = np.int32(lenY)
dx = np.float32(dx)
dy = np.float32(dy)
dx2 = np.float32(dx2)
dy2 = np.float32(dy2)
dt = np.float32(dt)
timesteps = np.int32(timesteps)
image_interval = np.int32(image_interval)

ker = SourceModule('''
/* Update the temperature values using five-point stencil */
__global__ void evolve_kernelCUDA(float *currdata, float *prevdata, float a, float dt, int nx, int ny,
                       float dx2, float dy2)
{

    /* Determine the temperature field at next time step
     * As we have fixed boundary conditions, the outermost gridpoints
     * are not updated. */
    int ind, iRight, iLeft, jUp, jDown;

    // CUDA threads are arranged in column major order; thus j index from x, i from y
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;        

    if (i > 0 && j > 0 && i < nx-1 && j < ny-1) {
        ind = i * ny + j;
        iRight = (i+1)  * (ny) + j;
        iLeft = (i - 1) * (ny ) + j;
        jUp = i * (ny) + j + 1;
        jDown = i * (ny) + j - 1;
        currdata[ind] = prevdata[ind] + a * dt *
	      ((prevdata[iRight] -2.0 * prevdata[ind] + prevdata[iLeft]) / dx2 +
	      (prevdata[jUp] - 2.0 * prevdata[ind] + prevdata[jDown]) / dy2);
    }
}
''')

evolve_kernel = ker.get_function('evolve_kernelCUDA')

def main():    
    # Set array size and set the interior value with Tguess
    field = np.empty((lenX, lenY))
    field.fill(Tguess)
    # Set Boundary condition
    field[(lenY - 1):, :] = Ttop
    field[:1, :] = Tbottom
    field[:, (lenX - 1):] = Tright
    field[:, :1] = Tleft
    
    print("field's size is ", field.size)
    print(field, "\n")

    field0 = field.copy()  # Array for field of previous time step
    print("field0's size is ", field0.size)
    print(field0, "\n")
    
    field = field.astype(np.float32)
    field0 = field0.astype(np.float32)
    
    write_field(field, 0)    
    blocksize = 32  # !< CUDA thread block dimension
    dimBlock = (blocksize, blocksize, 1)
    dimGrid = (int(lenX/blocksize+(0 if lenY % blocksize == 0 else 1)), 
               int(lenY/blocksize+(0 if lenX % blocksize == 0 else 1)), 
               1)
    print(dimBlock)
    print(dimGrid)
    print()
    
    # Allocate memory on device
    field_gpu = cuda.mem_alloc(field.nbytes)
    field0_gpu = cuda.mem_alloc(field0.nbytes)

    # Iterate
    t0 = time.time()
    for m in range(1, timesteps + 1):
        # Copy matrix to memory
        cuda.memcpy_htod(field_gpu, field)
        cuda.memcpy_htod(field0_gpu, field0)    
        
        evolve_kernel(field_gpu, field0_gpu, a, dt, lenX, lenY, dx2, dy2, block=dimBlock, grid=dimGrid)
        
        if (m % image_interval == 0):
            # Copy back the result
            cuda.memcpy_dtoh(field, field_gpu)
            write_field(field, m);
            # print(field, "\n")
        
        cuda.memcpy_dtoh(field, field_gpu)
        cuda.memcpy_dtoh(field0, field0_gpu)
        
        # Swap current field so that it will be used as previous for next iteration step
        tmp = field
        field = field0
        field0 = tmp
    
    t1 = time.time()
    print("Running time: {0}".format(t1 - t0))
    
    field_gpu.free()
    field0_gpu.free()

if __name__ == '__main__':
    main()

'''
https://github.com/csc-training/hpc-python/blob/master/mpi/heat-equation/solution/heat-p2p.py

基于 "D:\myCodes\HPCprojects\SourceCodes\parallel_python-master\mpi4py-heatequ2Dk.py" 进行修改！

2023年3月31日21:29:07 成功的版本！
原始版本在 "D:\myCodes\HPCprojects\SourceCodes\parallel_python-master\examplesHeatMPI\simpleFDM-PyCUDA.py"
文档资料 "D:\Local\++写书\18 高性能计算\HPCBook-MD\05-GPU-A-PyCUDA.md"

'''
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# evolve_kernel 来自 D:\myCodes\HPCprojects\heat-equation-main\cuda\core_cuda.cu