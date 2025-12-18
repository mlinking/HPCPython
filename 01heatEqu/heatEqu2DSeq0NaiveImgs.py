from __future__ import print_function
import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set the colormap
# plt.rcParams['image.cmap'] = 'BrBG'
plt.rcParams['image.cmap'] = 'jet' #you can try: colourMap = plt.cm.coolwarm
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
Tguess = 0.25

# Set array size and set the interior value with Tguess
T = np.empty((lenX, lenY))
T.fill(Tguess)

# Set Boundary condition
T[(lenY-1):, :] = Ttop
T[:1, :] = Tbottom
T[:, (lenX-1):] = Tright
T[:, :1] = Tleft
    
print("size is ",T.size)
print(T,"\n")

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
    plt.savefig('heat_Seq0NaiveImgs_{0:03d}.png'.format(step))

def main():
    write_field(T, 0)
    
    t0 = time.time()
    for m in range(1, timesteps + 1):
        T[1:-1, 1:-1] = T[1:-1, 1:-1] + a * dt * (
            (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] +
             T[:-2, 1:-1]) / dx2 +
            (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] +
             T[1:-1, :-2]) / dy2)
        if m % image_interval == 0:
            write_field(T, m)
    t1 = time.time()
    print("Running time: {0}".format(t1 - t0))    
    
if __name__ == '__main__':
    main()

'''
https://github.com/csc-training/hpc-python/blob/master/mpi/heat-equation/solution/heat-p2p.py

"D:\myCodes\HPCprojects\SourceCodes\parallel_python-master\mpi4py-heatequ2Dk.py"
'''


# for m in range(1, timesteps + 1):
    # for i in range(1, lenX-1):
        # for j in range(1, lenY-1):
            # T[i, j] = 0.25 * (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1])
