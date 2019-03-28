import bpy, bmesh
import numpy as np
import math
import mathutils
from numpy import linalg
from math import degrees
from mathutils import Vector

#Covariance matrix function
def covar(vertlist):
    dev = [[],[],[]]
    covarmat = [[],[],[]]
    for axis in range(3):
        for elem in vertlist[axis]:
            dev[axis].append(elem - np.mean(vertlist[axis]))
    print (dev)

    for i in range(3):
        np.dot(np.dot(dev[i], dev[(i+1)%3]), dev[(i+2)%3])
    #    for j in i:
