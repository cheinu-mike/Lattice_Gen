#to do: figure out how to get the magnitude of the vectors
    #get vertex coordinates with respect to new basis
    #get absolute maximum x' y' z' since those will be the maximum points and therefore will be the magnitude
    #create a global variable of the margin
#take the new values and then generate a lattice
#create buttons
#optional: the mean of the vertex points should be the volume and not at the mean of the vertex points
#optional: if volume is used then samples should be taken instead of taking the whole thing

margin = 0

import sys
sys.path.append('/home/cheinu/anaconda3/envs/cheinu/lib/python3.6/')

#%matplotlib inline
import bpy, bmesh
import numpy as np
import math
import mathutils
from numpy import linalg
from math import degrees
from mathutils import Vector
#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()

pi=math.pi
emptyvec = np.array([0,0,1])
emptyvec2 = np.array([0,1,0])

obj = bpy.data.objects['Cube']
mat = obj.matrix_world
objloc = list(mat.translation) #the location of the origin point as a list

#Get the vertices

if obj.mode == 'EDIT':
    bm = bmesh.from_edit_mesh(obj.data)
    verts = [list(vert.co) for vert in bm.verts]
    
else:
    verts = [list(vert.co) for vert in obj.data.vertices]
    matvert = [vert.co for vert in obj.data.vertices]

origin_verts = np.array(verts) + np.array([objloc[0],objloc[1], objloc[2]]) #global location of the verts
t_vert = np.transpose(origin_verts)                         #all the axis into single columns of X, Y, Z
lenlist = len(verts)                                        #length of list (number of vertices)
eigenvalue, eigenvector = linalg.eig(np.cov(t_vert))        #eigenvector and value

eigenvector = np.transpose(eigenvector)                     #fix eigenvector components

angle = math.acos(np.dot(emptyvec, eigenvector[0]))

qxyz = (np.cross(emptyvec,eigenvector[0]))*np.sin(angle/2)
qw = np.cos(angle/2)

v0 = Vector((0,0,1))
v1 = Vector((1,1,1))

#The mean of the vertex cloud
xmean = np.mean(t_vert[0]) #+ objloc[0]
ymean = np.mean(t_vert[1]) #+ objloc[1]
zmean = np.mean(t_vert[2]) #+ objloc[2]
#Calculate the magnitude

corverts = np.dot(eigenvector,t_vert)            #change of basis
cormean = np.dot(eigenvector, [xmean, ymean, zmean])
t_corverts = np.transpose(corverts)                     #transpose of the new vertex cloud position into PC axis

pcx = corverts[0] - cormean[0]
pcy = corverts[1] - cormean[1]
pcz = corverts[2] - cormean[2]

#print out stuff here to test
#print("number of vertices:  ", lenlist)              #print the number of vertices
#print(t_vert)
#print(origin_verts.tolist())
#print("Corrected vertices: ", t_corverts)
#print('covariance matrix: ', np.cov(t_vert))       #covariance matrix

print("cross product: \n",np.cross(emptyvec,eigenvector[0]))
print("Eigenvalues are: \n", eigenvalue)
print("Eigenvectors are: \n", eigenvector)        #each list represents an axis: [[x],[y],[z]]
print("Eigenvector length: \n", np.linalg.norm(eigenvector[0]))
print("Quaternion Axis:  \n", qxyz)
print("Quaternion Rotation:  \n", qw)
print("Angle between vectors:  \n", angle, '\n', 'in degrees: \n', math.degrees(angle),'\n')

print(max(abs(pcx)))
print(max(abs(pcy)))
print(max(abs(pcz)))

#print("Magnitude:  ", max(t_corverts[0]), max(t_corverts[1]) ,max(t_corverts[2]))
#print("The mean of the data points: ", np.mean(t_vert[0]))
#print(Vector(eigenvector[0]))

x,y,z = 0,1,2

t_eigen = np.transpose(eigenvector)
xeigen, yeigen, zeigen = Vector(eigenvector[0]), Vector(eigenvector[1]), Vector(eigenvector[2])

rot = v1.rotation_difference(v0).to_euler()

testrot = xeigen.rotation_difference(yeigen).to_euler()
#print([degrees(a) for a in rot])
#print([degrees(b) for b in testrot])

xangle = [math.atan(-eigenvector[y][i]/eigenvector[z][i]) for i in range(3)]
yangle = [math.atan(eigenvector[x][i]/eigenvector[z][i]) for i in range(3)]
zangle = [math.atan(eigenvector[y][i]/eigenvector[x][i]) for i in range(3)]
magnitude = max(t_corverts[0])
#print("degrees:  ", xangle)
#print("degrees:  ", yangle)
#print("degrees:  ", zangle)

#bpy.ops.object.empty_add(type='SINGLE_ARROW', 
#                        radius = 1, #magnitude,
#                        view_align=False, 
#                        location=(xmean, ymean, zmean), 
#                        rotation = (0, 0, 0),
#                        layers=(False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False))

quater = [qw]+list(qxyz)

bpy.context.object.rotation_mode = 'QUATERNION'
bpy.data.objects['Lattice'].rotation_quaternion = quater
bpy.context.object.location = [xmean,ymean,zmean]
bpy.context.object.scale = [2*max(abs(pcz)),2*max(abs(pcy)),2*max(abs(pcx))]
#print(verts)
#print(verts[1])
#meanx = np.mean(verts, axis = 0)
#print(meanx)