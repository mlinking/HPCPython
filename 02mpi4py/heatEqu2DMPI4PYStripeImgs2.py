from __future__ import print_function
import numpy as np
import time
from mpi4py import MPI

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set the colormap
# plt.rcParams['image.cmap'] = 'BrBG'
plt.rcParams['image.cmap'] = 'jet' #you can try: colourMap = plt.cm.coolwarm

# Basic parameters
a = 0.1  # Diffusion constant  #例如 Thermal diffusivity of steel, mm2.s-1
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

# Set Dimension and delta
lenX = lenY = 400 #we set it rectangular
delta = 1

# Boundary condition
Ttop = 100
Tbottom = 0
Tleft = 0
Tright = 30

# Initial guess of interior grid
Tguess = 0

# Set colour interpolation and colour map
colorinterpolation = 100
colourMap = plt.cm.jet #you can try: colourMap = plt.cm.coolwarm

# Set meshgrid
X, Y = np.meshgrid(np.arange(0, lenX), np.arange(0, lenY))

def init_fields(filename):
    # Read the initial temperature field from file
    field = np.loadtxt(filename)
    field0 = field.copy()  # Array for field of previous time step
    return field, field0

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

def write_field(field, step):
    plt.gca().clear()  
    plt.clf()    
    
    plt.figure(dpi=300)
    # Configure the contour
    plt.title("Contour of Temperature")
    plt.contourf(X, Y, field, colorinterpolation, cmap=colourMap)
    # Set Colorbar
    plt.colorbar()
    plt.axis('on')    
    plt.savefig('heat_MPIStride_{0:03d}.png'.format(step))

# MPI globals
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Up/down neighbouring MPI ranks
up = rank - 1
if up < 0:
    up = MPI.PROC_NULL
down = rank + 1
if down > size - 1:
    down = MPI.PROC_NULL

def evolve(u, u_previous, a, dt, dx2, dy2):
    """Explicit time evolution.
       u:            new temperature field
       u_previous:   previous field
       a:            diffusion constant
       dt:           time step
       dx2:          grid spacing squared, i.e. dx^2
       dy2:            -- "" --          , i.e. dy^2"""
    u[1:-1, 1:-1] = u_previous[1:-1, 1:-1] + a * dt * (
            (u_previous[2:, 1:-1] - 2 * u_previous[1:-1, 1:-1] +
             u_previous[:-2, 1:-1]) / dx2 +
            (u_previous[1:-1, 2:] - 2 * u_previous[1:-1, 1:-1] +
             u_previous[1:-1, :-2]) / dy2)
    u_previous[:] = u[:]

def exchange(field):
    # send down, receive from up
    sbuf = field[-2, :]
    rbuf = field[0, :]
    comm.Sendrecv(sbuf, dest=down, recvbuf=rbuf, source=up)
    # send up, receive from down
    sbuf = field[1, :]
    rbuf = field[-1, :]
    comm.Sendrecv(sbuf, dest=up, recvbuf=rbuf, source=down)

def main():
    # Read and scatter the initial temperature field
    if rank == 0:
        field = init_fields()
        shape = field.shape
        dtype = field.dtype
        comm.bcast(shape, root = 0)  # broadcast dimensions
        comm.bcast(dtype, root = 0)  # broadcast data type
    else:
        field = None
        shape = comm.bcast(None, 0)
        dtype = comm.bcast(None, 0)
    
    if shape[0] % size:
        raise ValueError('Number of rows in the temperature field (' \
                         + str(shape[0]) + ') needs to be divisible by the number ' \
                         + 'of MPI tasks (' + str(size) + ').')
    
    n = int(shape[0] / size)  # number of rows for each MPI task
    m = shape[1]  # number of columns in the field
    buff = np.zeros((n, m), dtype)
    comm.Scatter(field, buff, 0)  # scatter the data
    local_field = np.zeros((n + 2, m), dtype)  # need two ghost rows!
    local_field[1:-1, :] = buff  # copy data to non-ghost rows
    local_field0 = np.zeros_like(local_field)  # array for previous time step

    # Fix outer boundary ghost layers to account for aperiodicity?
    if True:
        if rank == 0:
            local_field[0, :] = local_field[1, :]
        if rank == size - 1:
            local_field[-1, :] = local_field[-2, :]
    local_field0[:] = local_field[:]

    # Plot/save initial field
    if rank == 0:
        write_field(field, 0)
    # Iterate
    t0 = time.time()
    for m in range(1, timesteps + 1):
        exchange(local_field0)
        evolve(local_field, local_field0, a, dt, dx2, dy2)
        if m % image_interval == 0:
            comm.Gather(local_field[1:-1, :], field, root=0)
            if rank == 0:
                write_field(field, m)
    t1 = time.time()
    
    # Plot/save final field
    comm.Gather(local_field[1:-1, :], field, root=0)
    if rank == 0:
        write_field(field, timesteps)
        print("Running time: {0}".format(t1 - t0))

if __name__ == '__main__':
    main()

'''
https://github.com/csc-training/hpc-python/blob/master/mpi/heat-equation/solution/heat-p2p.py

'''