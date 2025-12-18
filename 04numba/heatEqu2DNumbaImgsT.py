import numpy as np
from numba import jit, njit, prange
import timeit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Set the colormap
# plt.rcParams['image.cmap'] = 'BrBG'
plt.rcParams['image.cmap'] = 'jet' 
plt.figure(dpi=300)

# Set colour interpolation and colour map
colorinterpolation = 100
colourMap = plt.cm.jet #you can try: colourMap = plt.cm.coolwarm

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
lenX = lenY = 4000 #we set it rectangular
# Set meshgrid
X, Y = np.meshgrid(np.arange(0, lenX), np.arange(0, lenY))

def init_fields():
    # Set array size and set the interior value with Tguess
    field = np.empty((lenX, lenY))
    field.fill(Tguess)

    # Set Boundary condition
    field[(lenY-1):, :] = Ttop
    field[:1, :] = Tbottom
    field[:, (lenX-1):] = Tright
    field[:, :1] = Tleft
    return field

def write_field(field, step):
    # plt.gca().clear()
    # plt.cla()
    plt.clf()
    
    # Configure the contour
    plt.title("Contour of Temperature")
    plt.contourf(X, Y, field, colorinterpolation, cmap=colourMap)
    # Set Colorbar
    plt.colorbar()
    plt.axis('on')    
    plt.savefig('heat_Numba4K_{0:03d}.png'.format(step))
    
# nopython = True选项要求完全编译该函数（以便完全删除Python解释器调用），否则会引发异常。
# 这些异常通常表示函数中需要修改的位置，以实现优于Python的性能。强烈建议您始终使用nopython = True。
# @jit(nopython=True)
# @numba.njit(cache=True, parallel=True)
# @jit(nopython = True, parallel = True, nogil = True)
# @jit
# @njit(fastmath=True)
@njit(fastmath=True)
def evolve(u, lenX, lenY, a, dt, dx2, dy2):
    for i in range(1, lenX-1, 1):
        for j in range(1, lenY-1, 1):
            u[i, j] = u[i, j] + a * dt * (
                (u[i+1, j] - 2 * u[i, j] +
                 u[i-1, j]) / dx2 +
                (u[i, j+1] - 2 * u[i, j] +
                 u[i, j-1]) / dy2)

def main():
    field = init_fields()
    write_field(field, 0)
    
    print("size is ",field.size)
    print(field,"\n")

    # Iteration (We assume that the iteration is convergence in maxIter = 500)
    print("Please wait for a moment")

    starting_time = timeit.default_timer()
    for iteration in range(0, timesteps+1):
        evolve(field, lenX, lenY, a, dt, dx2, dy2)
        if iteration % image_interval == 0:
            write_field(field, iteration)

    print("Iteration finished. {} Seconds for Time difference:".format(timeit.default_timer() - starting_time))

if __name__ == '__main__':
    main()

""" https://www.codeproject.com/Articles/1087025/Using-Python-to-Solve-Computational-Physics-Proble
# D:\My7\MyProjects\Memoirs\2018-12-18-Heat Transfer with Python.docx
# 2019年1月8日23:03:49

2023年4月27日15:56:13
D:\myCodes\HPCprojects\SourceCodes\parallel_python-master\examplesHeatMPI\simpleFDM-Numba1.py
"""
# Simple Numerical Laplace Equation Solution using Finite Difference Method