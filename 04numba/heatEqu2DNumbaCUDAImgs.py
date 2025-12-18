from __future__ import print_function
# Simple Numerical Laplace Equation Solution using Finite Difference Method
import numpy as np
import timeit

from numba import cuda # 从numba调用cuda

import matplotlib
import matplotlib.pyplot as plt

# Set the colormap
# plt.rcParams['image.cmap'] = 'BrBG'
plt.rcParams['image.cmap'] = 'jet' #you can try: colourMap = plt.cm.coolwarm
# plt.figure(dpi=300)

# Set colour interpolation and colour map
colorinterpolation = 100
colourMap = plt.cm.jet #you can try: colourMap = plt.cm.coolwarm

# Basic parameters
a = 0.1  # Diffusion constant  #例如 Thermal diffusivity of steel, mm2.s-1
timesteps = 10000  # Number of time-steps to evolve system
image_interval = 400

# Grid spacings
dx = 0.01
dy = 0.01
dx2 = dx ** 2
dy2 = dy ** 2

# For stability, this is the largest interval possible
# for the size of the time-step:
dt = dx2 * dy2 / (2 * a * (dx2 + dy2))

# Set Dimension 
lenX = lenY = 400 #we set it rectangular

# Set meshgrid
X, Y = np.meshgrid(np.arange(0, lenX), np.arange(0, lenY))

# Boundary condition
Ttop = 100
Tbottom = 0
Tleft = 0
Tright = 30

# Initial guess of interior grid
Tguess = 0


# timesteps = np.int32(timesteps)
# image_interval = np.int32(image_interval)

# 2023年5月8日10:39:11 网上看到的 生成一个元素的np array
# a_arr = np.zeros(1, dtype=np.float32) #常数转矩阵
# a_arr[0] = a
# 其实 a = np.array([0.01],np.float32) 也就行了
a_cuda = np.array([a], dtype = np.float32)
lenX_cuda = np.array([lenX], dtype = np.int32)
lenY_cuda = np.array([lenY], dtype = np.int32)
dx_cuda = np.array([dx], dtype = np.float32)
dy_cuda = np.array([dy], dtype = np.float32)
dx2_cuda = np.array([dx2], dtype = np.float32)
dy2_cuda = np.array([dy2], dtype = np.float32)
dt_cuda = np.array([dt], dtype = np.float32)


def init_fields():
    # Set array size and set the interior value with Tguess
    field = np.zeros((lenX, lenY), dtype=np.float32)
    field.fill(Tguess)

    # Set Boundary condition
    field[(lenY-1):, :] = Ttop
    field[:1, :] = Tbottom
    field[:, (lenX-1):] = Tright
    field[:, :1] = Tleft
    
    print("size is ",field.size)
    print(field,"\n")
    
    field0 = field.copy()  # Array for field of previous time step   
    return field, field0

@cuda.jit
def evolve(u, u_previous, a, dt, xlen, ylen, dx2, dy2):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if (i > 0 and j > 0 and i < xlen[0]-1 and j < ylen[0]-1):
        u[i][j] = u_previous[i][j] + a[0] * dt[0]*((u_previous[i+1][j] - 2.0 * u_previous[i][j] + u_previous[i-1][j])/dx2[0] + 
            (u_previous[i][j+1] - 2.0 * u_previous[i][j] + u_previous[i][j-1])/dy2[0])


def main():
    field, field0 = init_fields()

    blocksize = 32  # !< CUDA thread block dimension
    dimBlock = (blocksize, blocksize)
    # CUDA threads are arranged in column major order; thus make ny x nx grid
    # dimGrid = ((lenY + 2 + blocksize - 1) // blocksize,
              # (lenX + 2 + blocksize - 1) // blocksize,
              # 1)
    
    
    dimGrid = (int(lenX/blocksize+(0 if lenY % blocksize == 0 else 1)), 
               int(lenY/blocksize+(0 if lenX % blocksize == 0 else 1)))
    print(dimBlock)
    print(dimGrid)
    
    starting_time = timeit.default_timer()
    for m in range(1, timesteps + 1):
        field_device = cuda.to_device(field)
        field0_device = cuda.to_device(field0)
        a_device  = cuda.to_device(a_cuda)
        dt_device  = cuda.to_device(dt_cuda)
        lenX_device  = cuda.to_device(lenX_cuda)
        lenY_device  = cuda.to_device(lenY_cuda)
        dx2_device  = cuda.to_device(dx2_cuda)
        dy2_device  = cuda.to_device(dy2_cuda)
        
        cuda.synchronize()
        evolve[dimGrid,dimBlock](field_device, field0_device, a_device, dt_device, lenX_device, lenY_device, dx2_device, dy2_device)
        cuda.synchronize()
        
        field = field0_device.copy_to_host()
        field0 = field_device.copy_to_host()
        
    # print("Iteration finished")
    print("Iteration finished. {} Seconds for Time difference:".format(timeit.default_timer() - starting_time))
    
    # Configure the contour
    plt.title("Contour of Temperature")
    plt.contourf(X, Y, field, colorinterpolation, cmap=colourMap)
    # Set Colorbar
    plt.colorbar()
    plt.axis('on') 
    # Show the result in the plot window
    plt.show()

if __name__ == '__main__':
    main()

'''
https://github.com/csc-training/hpc-python/blob/master/mpi/heat-equation/solution/heat-p2p.py

"D:\myCodes\HPCprojects\SourceCodes\parallel_python-master\mpi4py-heatequ2Dk.py"

2023年5月8日10:27:57 直接使用数字测试，也就通过了！

'''
