from __future__ import print_function
import numpy as np
import time
from mpi4py import MPI
import h5py

# Basic parameters
a = 0.1  # Diffusion constant  #例如 Thermal diffusivity of steel, mm2.s-1
timesteps = 100  # Number of time-steps to evolve system
image_interval = 10  # Write frequency for png files

# Grid spacings
dx = 0.01
dy = 0.01
dx2 = dx ** 2
dy2 = dy ** 2

# For stability, this is the largest interval possible for the size of the time-step:
dt = dx2 * dy2 / (2 * a * (dx2 + dy2))

# MPI globals
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Up/down neighbouring MPI ranks
upNeigh = rank - 1
if upNeigh < 0:
    upNeigh = MPI.PROC_NULL
downNeigh = rank + 1
if downNeigh > size - 1:
    downNeigh = MPI.PROC_NULL

neigh = [downNeigh, upNeigh]

def evolve(u, a, dt, dx2, dy2, neigh):
    # neigh = [downNeigh, upNeigh]
    up = 1
    down = u.shape[0]-1
    right = u.shape[1]-1
    left = 1
    
    validDirections = np.where(np.array(neigh) ==-1)
    for direction in validDirections[0]:
        if direction == 0: # Down
            down = u.shape[0]-2
        elif direction == 1: # Up
            up = 2
    
    u[up:down, left:right] = u[up:down, left:right] + a * dt * (
            (u[up+1:down+1, left:right] - 2 * u[up:down, left:right] +
             u[up-1:down-1, left:right]) / dx2 +
            (u[up:down, left+1:right+1] - 2 * u[up:down, left:right] +
             u[up:down, left-1:right-1]) / dy2)

def exchange(localfield):
    # send downNeigh, receive from upNeigh
    sbuf = localfield[-2, :]
    rbuf = localfield[-1, :]
    comm.Sendrecv(np.array(sbuf), dest=downNeigh, recvbuf=rbuf, source=downNeigh)
    # send upNeigh, receive from downNeigh
    sbuf = localfield[1, :]
    rbuf = localfield[0, :]
    comm.Sendrecv(np.array(sbuf), dest=upNeigh, recvbuf=rbuf, source=upNeigh)

filename = 'heat2DEqu_h5py.h5'
dataSetName = 'Heat2D'
dataSize = 10000
stripeSize = 250
Nmpiprocesses = dataSize//stripeSize

tempDir = './heatEqu2DMPI4PY_H5PY_StripeImgs'
tempFile = 'heat2DEqu_h5pyTMP'

def main():
    # 打开HDF5文件
    with h5py.File(filename, 'r') as file:
        # 获取数据集
        dataset = file[dataSetName]
        
        if dataset.shape[0] % size:
            raise ValueError('Number of rows in the temperature field (' \
                             + str(shape[0]) + ') needs to be divisible by the number ' \
                             + 'of MPI tasks (' + str(size) + ').')
        
        # 确定数据块大小
        chunk_size = (dataset.shape[0] // size, dataset.shape[1])
        
        # 计算每个进程的切片范围
        if rank == size - 1:
            # 最后一个进程读取剩余的所有数据
            start_index = rank * chunk_size[0]
            end_index = dataset.shape[0]
        else:
            start_index = rank * chunk_size[0]
            end_index = start_index + chunk_size[0]
        
        local_field = np.zeros((end_index -start_index + 2, dataset.shape[1]), dtype = float)  # need two ghost rows!
        local_field[1:-1, :] = dataset[start_index:end_index, :]  # copy data to non-ghost rows

        # Iterate
        t0 = time.time()
        for m in range(1, timesteps + 1):
            exchange(local_field)
            evolve(local_field, a, dt, dx2, dy2, neigh)
            if m % image_interval == 0:
                # 当运行到指定的次数后，每个 MPI 进程将自己的数据写入到一个HDF5 文件中 - local_field[1:-1, :]
                with h5py.File(tempDir+'/'+tempFile+'-'+str(m)+'-'+str(rank)+'.h5', 'w') as tmpfile:
                    tmpfile.create_dataset('rank-'+str(rank),data = local_field[1:-1, :])
                # pass
                
        t1 = time.time()
        
        # Plot/save final field
        # comm.Gather(local_field[1:-1, :], field, root=0)
        if rank == 0:
            # write_field(field, timesteps)
            print("Running time: {0}".format(t1 - t0))

if __name__ == '__main__':
    main()

'''
2024年6月11日20:04:24

来自 Kimi 
D:\myCodes\HPCbook\02mpi4py\heatEqu2DMPI4PYStripeImgs2T.py

mpiexec -n 40 python heatEqu2DMPI4PY_H5PY_StripeImgs2T.py

'''