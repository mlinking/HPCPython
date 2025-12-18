from __future__ import print_function
import numpy as np
import timeit

# =============matplotlib 配置 =========================
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

# Set Dimension 
lenX = lenY = 400 #we set it rectangular

# Set meshgrid
X, Y = np.meshgrid(np.arange(0, lenX), np.arange(0, lenY))

# =============参数的配置 =====================

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
Tguess = 0

def g(x):
    print(x)

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
    # plt.gca().clear()
    # plt.cla()
    plt.clf()    
    
    # Configure the contour
    plt.title("Contour of Temperature")
    plt.contourf(X, Y, field, colorinterpolation, cmap=colourMap)
    # Set Colorbar
    plt.colorbar()
    plt.axis('on')    
    plt.savefig('heat_Spark6_{0:03d}.png'.format(step))   

# ============================================================
# import pyspark
# sc = pyspark.SparkContext.getOrCreate()
import os

from pyspark import SparkConf, SparkContext

conf = SparkConf()
conf.set("spark.driver.maxResultSize", "8g")
conf.set("spark.driver.memory", "12g")
sc = SparkContext(conf=conf).getOrCreate()


def g(x):
    print(x)

def tupleIndex(tup: tuple) -> list:
    """
    主要作用是把每个值打上行列index
    """
    Array, row_index = tup
    return [[row_index, col_index, value] for col_index, value in enumerate(Array)]

def merge_by_index(tup: tuple):
    """
    按row_index 把对应数据放入对应位置
    """
    row_index, data_list = tup
    res = [0]*len(data_list)
    for col_index, value in data_list:
        res[col_index] = value
    return (row_index, np.array(res, dtype = float))

internalSelf = 1.0 - 2.0*a*dt/dx2 - 2.0*a*dt/dy2
diff4DirectionsDx = a*dt/dx2
diff4DirectionsDy = a*dt/dy2

def diffuse(tup: tuple) -> tuple:
    # print("in diffuse - ", tup)
    # print("in diffuse - tup[0]", tup[0], " - tup[1] ", tup[1])
    res = []
    if tup[0][0] > 1 and tup[0][0] < lenX-2 and tup[0][1] > 1 and tup[0][1] < lenY-2: # 两圈之内的 - 传5个值：自己的+4个方向的
        res.append(((tup[0][0],tup[0][1]), tup[1]*internalSelf))
        # 四个方向都辐射
        res.append(((tup[0][0]+1,tup[0][1]), tup[1]*diff4DirectionsDx))
        res.append(((tup[0][0]-1,tup[0][1]), tup[1]*diff4DirectionsDx))
        res.append(((tup[0][0],tup[0][1]+1), tup[1]*diff4DirectionsDy))
        res.append(((tup[0][0],tup[0][1]-1), tup[1]*diff4DirectionsDy))
    else:
        if tup[0][0] > 0 and tup[0][0] < lenX-1 and tup[0][1] > 0 and tup[0][1] < lenY-1: # 第二圈的点
            res.append(((tup[0][0],tup[0][1]), tup[1]*internalSelf))
            if tup[0][0] == 1:
                res.append(((tup[0][0]+1,tup[0][1]), tup[1]*diff4DirectionsDx))
                if tup[0][1] == 1:
                    res.append(((tup[0][0],tup[0][1]+1), tup[1]*diff4DirectionsDy))
                elif tup[0][1] == lenY-2:
                    res.append(((tup[0][0],tup[0][1]-1), tup[1]*diff4DirectionsDy))
                else:
                    res.append(((tup[0][0],tup[0][1]-1), tup[1]*diff4DirectionsDy))
                    res.append(((tup[0][0],tup[0][1]+1), tup[1]*diff4DirectionsDy))
            elif tup[0][0] == lenX-2:
                res.append(((tup[0][0]-1,tup[0][1]), tup[1]*diff4DirectionsDx))
                if tup[0][1] == 1:
                    res.append(((tup[0][0],tup[0][1]+1), tup[1]*diff4DirectionsDy))
                elif tup[0][1] == lenY-2:
                    res.append(((tup[0][0],tup[0][1]-1), tup[1]*diff4DirectionsDy))
                else:
                    res.append(((tup[0][0],tup[0][1]-1), tup[1]*diff4DirectionsDy))
                    res.append(((tup[0][0],tup[0][1]+1), tup[1]*diff4DirectionsDy))
            else:
                res.append(((tup[0][0]-1,tup[0][1]), tup[1]*diff4DirectionsDx))
                res.append(((tup[0][0]+1,tup[0][1]), tup[1]*diff4DirectionsDx))
                if tup[0][1] == 1:
                    res.append(((tup[0][0],tup[0][1]+1), tup[1]*diff4DirectionsDy))
                elif tup[0][1] == lenY-2:
                    res.append(((tup[0][0],tup[0][1]-1), tup[1]*diff4DirectionsDy))
        else: # 最外一圈的点了
            res.append(((tup[0][0],tup[0][1]), tup[1]))
            if (tup[0][0],tup[0][1]) not in [(0, 0),(0, lenY-1),(lenX-1,0),(lenX-1,lenY-1)]:
                if tup[0][0] == 0:
                    res.append(((tup[0][0]+1,tup[0][1]), tup[1]*diff4DirectionsDx))
                elif tup[0][0] == lenX-1:
                    res.append(((tup[0][0]-1,tup[0][1]), tup[1]*diff4DirectionsDx))
                elif tup[0][1] == 0:
                    res.append(((tup[0][0],tup[0][1]+1), tup[1]*diff4DirectionsDy))
                elif tup[0][1] == lenY-1:
                    res.append(((tup[0][0],tup[0][1]-1), tup[1]*diff4DirectionsDy))
    # print("in diffuse - ", res)
    return res

def main():
    
    mat = init_fields()
    write_field(mat, 0)
    
    rdd = sc.parallelize(mat)
    
    # rdd.foreach(g)
    # print()
    starting_time = timeit.default_timer()
    matrix = rdd.zipWithIndex().flatMap(lambda x: tupleIndex(x)).map(lambda x: ((x[0],x[1]), x[2]))
    # print("matrix - ")
    # matrix.foreach(g)
    # print()
    
    # for m in range(1, 5+1):
    for m in range(1, timesteps + 1):
        diffused = matrix.flatMap(lambda x:diffuse(x))
        # print("diffused - ", m)
        # diffused.foreach(g)
        # print()
        matrixCollected = diffused.reduceByKey(lambda x,y: x+y).collect()
        # print("matrixCollected - ", matrixCollected)
        matrix = sc.parallelize(matrixCollected)
        # print(" - ", m)
        # matrix.foreach(g)
        # print()
        
        if m % image_interval == 0:
        # if m % 2 == 0:
            # print("in if m % image_interval == 0\n")
            matrix_PRT = matrix.map(lambda x: (x[0][0], (x[0][1], x[1]))).groupByKey().map(lambda x:merge_by_index(x)).sortBy(lambda x: x[0], ascending = True).mapValues(list).map(lambda x: x[1]).collect()
            # print("matrix_PRT - ", m, "\n", matrix_PRT)
            # print()
            
            rdd7 = np.asarray(matrix_PRT)
            write_field(rdd7, m)
    
    # print("Iteration finished")
    print("Iteration finished. {} Seconds for Time difference:".format(timeit.default_timer() - starting_time))
    
    sc.stop()    

if __name__ == "__main__":
    main()


"""
基于两个 准备的代码

"D:\myCodes\HPCbook\05pyspark\sparkTest1NumpyMattoRDD9.py"
"D:\myCodes\HPCbook\05pyspark\sparkTest1NumpyMattoRDD10.py"


"""
