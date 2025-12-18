from __future__ import print_function
import h5py
import natsort

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Set the colormap
# plt.rcParams['image.cmap'] = 'BrBG'
plt.rcParams['image.cmap'] = 'jet' 
# Set colour interpolation and colour map
colorinterpolation = 100
colourMap = plt.cm.jet #you can try: colourMap = plt.cm.coolwarm

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
    plt.savefig('heat_MPIStride_HDF5Vis_{0:03d}.png'.format(step))

resultFile = 'heatEqu2DMPI4PY_H5PY_StripeImgs2T.h5'
dataWidth = 400
# Set meshgrid
X, Y = np.meshgrid(np.arange(0, dataWidth), np.arange(0, dataWidth))

with h5py.File(resultFile, 'r') as file:
    datasets = natsort.natsorted(list(file.keys()))
    for datasetName in datasets:
        dataset = file[datasetName]
        span = dataset.shape[0]//dataWidth
        distilledData = dataset[0:dataset.shape[0]:span,0:dataset.shape[1]:span]
        distilledData[dataWidth-1,:] = dataset[dataset.shape[0]-1,2]
        distilledData[:,dataWidth-1] = dataset[2,dataset.shape[0]-1]
        write_field(distilledData, int(datasetName))

    
"""
借鉴了 D:\myCodes\HPCbook\11-HUGE Data\heat2DEqu_h5py_Show.py

"""