#to do: figure out how to get the magnitude of the vectors
    #Fix quaternion rotation to take into account y and z of eigenvectors
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

#testing section
#==========================
class HelloWorldOperator(bpy.types.Operator):
    bl_idname = "wm.hello_world"
    bl_label = "Minimal Operator"

    def execute(self, context):
        print("Hello World")
        return {'FINISHED'}

bpy.utils.register_class(HelloWorldOperator)

# test call to the newly defined operator
bpy.ops.wm.hello_world()

class HelloWorldPanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_hello_world"
    bl_label = "Hello World"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "object"

    def draw(self, context):
        self.layout.label(text="Hello World")


bpy.utils.register_class(HelloWorldPanel)

#=========

pi=math.pi

#testing matrices delete later
emptyvec = np.array([0,0,1])
emptyvec2 = np.array([0,1,0])

#*******SELECT THE OBJECT YOU WANT******
obj = bpy.data.objects['Cube']
#***************************************
mat = obj.matrix_world
objloc = list(mat.translation) #the location of the origin point as a list

#Get the vertices

if obj.mode == 'EDIT':
    bm = bmesh.from_edit_mesh(obj.data)
    verts = [list(vert.co) for vert in bm.verts]
    
else:
    verts = [list(vert.co) for vert in obj.data.vertices]
    matvert = [vert.co for vert in obj.data.vertices]

#returns list of the selected objects (bpy.context.selected_objects)
def get_objname(selected_objects):
    return [item.name for item in selected_objects]
        
get_objname(bpy.context.selected_objects)

#Extract the local vertex coordinates of the object
def get_localvert_coor(active_object):      
    if active_object.mode == 'EDIT':
        bm = bmesh.from_edit_mesh(active_object.data)
        verts = [list(vert.co) for vert in bm.verts]
        return verts
    else:
        verts = [list(vert.co) for vert in active_object.data.vertices]
        return verts

#Extract the vertex coordinates with respect to PC axis
def get_PC_coor(verts, vert_mean, eigenvector):
    PC_verts = np.dot(eigenvector, verts)
    PC_mean = np.dot(eigenvector, vert_mean)
    
    return PC_verts - PC_mean
    
#extract quaternions from vector positions
def get_quaternion(vector_1, vector_2, rotation_vector): #use the crossproduct for rotation_vector first 
    angle = math.acos(np.dot(vector_1, vector_2)) #breakdown of dot product of the vectors which gives the angle
    qxyz = (rotation_vector)*np.sin(angle/2)
    qw = np.cos(angle/2)
    quaternion = [qw]+list(qxyz)
    
    return quaternion
#===============================
class PCA:
    def __init__(self, selected_object):   #make sure selected_object = bpy.data.objects['NAME'] or bpy.context.selected_objects
        self.objname = selected_object.name
        self.localverts = get_localvert_coor(selected_object)
        self.globalverts = np.array(self.localverts) + np.array(selected_object.matrix_world.translation)
        t_verts = np.transpose(self.globalverts)    #Columns of X, Y, Z
        self.covarmat = (np.cov(t_verts))           #covariance matrix
        self.eigenvalue, self.eigenvector = linalg.eig(np.cov(t_verts))
        self.eigenvector = np.transpose(self.eigenvector)
        self.xmean = np.mean(t_verts[0])
        self.ymean = np.mean(t_verts[1])
        self.zmean = np.mean(t_verts[2])
    def testprint(self):
        print("this is printing from inside the class", self.xmean)
        
    def create_lattice(self):
        lattice_name = 'PCA_' + str(self.name) + '_lattice'
        bpy.ops.object.add(type='LATTICE', view_align=False, enter_editmode=False, location=(self.xmean, self.ymean, self.zmean))
        bpy.data.objects['Lattice'].name = lattice_name
        
        pass
def global_vert_position(vert_array, object_coor):
    global_verts  = np.array(vert_array) + np.array(object_coor)
    return global_verts

PCA(obj).objname
#===============================    
origin_verts = np.array(verts) + np.array([objloc[0],objloc[1], objloc[2]]) #global location of the verts
origin_verts = global_vert_position(verts, objloc)

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

'''
print("cross product: \n",np.cross(emptyvec,eigenvector[0]))
print("Eigenvalues are: \n", eigenvalue)
print("Eigenvectors are: \n", eigenvector)        #each list represents an axis: [[x],[y],[z]]
print("Eigenvector length: \n", np.linalg.norm(eigenvector[0]))
print("Quaternion Axis:  \n", qxyz)
print("Quaternion Rotation:  \n", qw)
print("Angle between vectors:  \n", angle, '\n', 'in degrees: \n', math.degrees(angle),'\n')
'''
'''
print(max(abs(pcx)))
print(max(abs(pcy)))
print(max(abs(pcz)))
'''
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

quater = [qw]+list(qxyz)

for obj in bpy.data.objects:
    obj.select_set(False)
bpy.data.objects['Cube'].select_set(False)
bpy.data.objects['Lattice'].select_set(True)
bpy.data.objects['Lattice'].rotation_mode = 'QUATERNION'
bpy.data.objects['Lattice'].rotation_quaternion = quater
bpy.data.objects['Lattice'].location = [xmean,ymean,zmean]
bpy.data.objects['Lattice'].scale = [2*max(abs(pcz)),2*max(abs(pcy)),2*max(abs(pcx))]