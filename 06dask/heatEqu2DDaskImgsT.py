from __future__ import print_function
import numpy as np
import time

import dask

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(dpi=300)
# Set the colormap
# plt.rcParams['image.cmap'] = 'BrBG'
plt.rcParams['image.cmap'] = 'jet' #you can try: colourMap = plt.cm.coolwarm
# Set colour interpolation and colour map
colorinterpolation = 100
colourMap = plt.cm.jet #you can try: colourMap = plt.cm.coolwarm

# Set Dimension 
lenX = lenY = 400 #we set it rectangular

# Set meshgrid
X, Y = np.meshgrid(np.arange(0, lenX), np.arange(0, lenY))

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
Ttop = 100.
Tbottom = 0.
Tleft = 0.
Tright = 30.
# Initial guess of interior grid
Tguess = 0.25

def init_fields():
    # Set array size and set the interior value with Tguess
    field = np.empty((lenX, lenY))
    field.fill(Tguess)

    # Set Boundary condition
    field[(lenY-1):, :] = Ttop
    field[:1, :] = Tbottom
    field[:, (lenX-1):] = Tright
    field[:, :1] = Tleft
        
    print("size is ",field.size)
    print(field,"\n")

    return field

@dask.delayed
def evolve(u,  a, dt, dx2, dy2):
    u[1:-1, 1:-1] = u[1:-1, 1:-1] + a * dt * (
            (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] +
             u[:-2, 1:-1]) / dx2 +
            (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] +
             u[1:-1, :-2]) / dy2)

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
    plt.savefig('heat_DaskBook_{0:03d}.png'.format(step))

def main():
    field = init_fields()
    write_field(field, 0)
    
    t0 = time.time()
    for m in range(1, timesteps + 1):
        dask.compute(evolve(field, a, dt, dx2, dy2))
        if m % image_interval == 0:
            write_field(field, m)
    t1 = time.time()
    print("Running time: {0}".format(t1 - t0))
    
if __name__ == '__main__':
    main()

'''

'''