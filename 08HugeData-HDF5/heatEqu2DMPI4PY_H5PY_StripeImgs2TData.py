from __future__ import division, print_function
import numpy as np

import h5py

# Boundary condition
Ttop = 100.
Tbottom = 0.
Tleft = 0.
Tright = 30.
# Initial guess of interior grid
Tguess = 0.25

dataFile = 'heat2DEqu_h5py.h5'
dataSet = 'Heat2D'

dataSize = 10000
stripeSize = 250
N = dataSize//stripeSize
lenX = lenY = dataSize

# 行切 - 对输入的二维数据 (x,y)，分 n 份添加入 file_h5，要求满足边界条件
def init_fields(file_h5, x, y, top = Ttop, bottom = Tbottom, left = Tleft, right = Tright, guess = Tguess, n = N):
    # 首先要判断 x能整除n，否则提示 "应整除" (此处实验能够整除，不能整除的情况读者可自行添加)
    while i < n:
        if i ==0:
            with h5py.File(file_h5, "w") as h5file:
                field = np.zeros((stripeSize, dataSize), dtype = np.float32)
                field.fill(guess)
                field[0, :] = bottom
                field[:, (y-1)] = right
                field[:, :1] = left
                h5file.create_dataset('Heat2D', (stripeSize, y), data = field, maxshape=(None, y))
        elif i == n-1:
            with h5py.File(file_h5, "a") as h5file:
                field = np.zeros((stripeSize, dataSize), dtype = np.float32)
                field.fill(guess)
                field[(stripeSize-1), :] = top
                field[:, (y-1):] = right
                field[:, :1] = left
                data = h5file['Heat2D']
                data.resize(data.shape[0]+stripeSize, axis=0)
                data[i*stripeSize:(i+1)*stripeSize,:] = field
        else:
            with h5py.File(file_h5, "a") as h5file:
                field = np.zeros((stripeSize, dataSize), dtype = np.float32)
                field.fill(guess)
                field[:, (y-1):] = right
                field[:, :1] = left
                data = h5file['Heat2D']
                data.resize(data.shape[0]+stripeSize, axis=0)
                data[i*stripeSize:(i+1)*stripeSize,:] = field            
        print(i)
        i=i+1
        
if __name__ == '__main__':
    init_fields(dataFile, lenX, lenY)

"""
构建一个 10000x10000的2维热传导计算的数据集

python heat2DEqu_h5py.py


"""