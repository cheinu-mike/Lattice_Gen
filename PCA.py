#top priority: align y to eigen_y
    #Fix quaternion rotation to take into account y and z of eigenvectors
    #create a global variable of the margin
#take the new values and then generate a lattice
#create buttons
#optional: the mean of the vertex points should be the volume and not at the mean of the vertex points
#optional: if volume is used then samples should be taken instead of taking the whole thing
#optional: multi object functionality

margin = 0

import sys

import bpy, bmesh
import numpy as np
import math
import mathutils
from numpy import linalg
from math import degrees
from mathutils import Vector

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
#emptyvec2 = np.array([0,1,0])

#*******SELECT THE OBJECT YOU WANT******
obj = bpy.data.objects['Cube']
obj2 = bpy.data.objects['Cube.001']
obj3 = bpy.data.objects['Cube.002']
#***************************************


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
def get_PC_coor(verts, vert_mean, eigenmatrix):
    PC_verts = np.dot(eigenmatrix, verts)
    PC_mean = np.dot(eigenmatrix, vert_mean)
    
    PC_verts = np.transpose(PC_verts) - PC_mean
    
    return np.transpose(PC_verts)
    
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
        self.basex = np.array([1,0,0])
        self.basey = np.array([0,1,0])
        self.basez = np.array([0,0,1])
        self.objname = selected_object.name
        self.localverts = get_localvert_coor(selected_object)
        self.globalverts = np.array(self.localverts) + np.array(selected_object.matrix_world.translation)
        self.t_verts = np.transpose(self.globalverts)    #Columns of X, Y, Z
        self.covarmat = (np.cov(self.t_verts))           #covariance matrix
        self.eigenvalue, self.eigenvector = linalg.eig(self.covarmat)
        self.eigenvector = np.transpose(self.eigenvector)
        self.xmean = np.mean(self.t_verts[0])
        self.ymean = np.mean(self.t_verts[1])
        self.zmean = np.mean(self.t_verts[2])
        self.mean = [self.xmean, self.ymean, self.zmean]
        self.pcx = get_PC_coor(self.t_verts, self.mean, self.eigenvector)[0]
        self.pcy = get_PC_coor(self.t_verts, self.mean, self.eigenvector)[1]
        self.pcz = get_PC_coor(self.t_verts, self.mean, self.eigenvector)[2]
        
    def testprint(self):
        print("this is printing from inside the class", self.xmean)
        
    def create_lattice(self, marginx = 0, marginy = 0, marginz = 0):
        for obj in bpy.data.objects:
            obj.select_set(False)
        
        lattice_name = 'PCA_' + str(self.objname) + '_lattice'
        
        #if any(obj == bpy.data.objects[lattice_name] for obj in bpy.data.objects):
        
        bpy.ops.object.add(type='LATTICE', view_align=False, enter_editmode=False, location=(self.xmean, self.ymean, self.zmean))
        bpy.data.objects['Lattice'].name = lattice_name
            
        quat_z = get_quaternion(self.basez, self.eigenvector[0], np.cross(self.basez,self.eigenvector[0]))
        quat_y = get_quaternion(self.basey, self.eigenvector[1], self.eigenvector[0])
        quat_x = get_quaternion(self.basex, self.eigenvector[2], self.eigenvector[1])
        
        #self.eigenvector = [self.eigenvector[1], self.eigenvector[2], self.eigenvector[0]]
        #quat_z = mathutils.Matrix(self.eigenvector).to_quaternion()
        
        print("Eigenvectors are: \n", self.eigenvector) 
        print("Quaternions are: \n", quat_z)
        
        '''
        t = np.trace(self.eigenvector)
        r = math.sqrt(1+t)
        s = 1/(2*r)
        qw = r/2
        qx = (self.eigenvector[2][1] - self.eigenvector[1][2]) * s
        qy = (self.eigenvector[0][2] - self.eigenvector[2][0]) * s
        qz = (self.eigenvector[1][0] - self.eigenvector[0][1]) * s
        
        signx = (self.eigenvector[2][1] - self.eigenvector[1][2])/abs(self.eigenvector[2][1] - self.eigenvector[1][2])
        signy = (self.eigenvector[0][2] - self.eigenvector[2][0])/abs(self.eigenvector[0][2] - self.eigenvector[2][0])
        signz = (self.eigenvector[1][0] - self.eigenvector[0][1])/abs(self.eigenvector[1][0] - self.eigenvector[0][1])
        
        qw = math.sqrt(np.trace(self.eigenvector) + 1)/2

        qx = signx * abs(np.sqrt(1 + self.eigenvector[0][0] - self.eigenvector[1][1] - self.eigenvector[2][2])/2)
        qy = signy * abs(np.sqrt(1 - self.eigenvector[0][0] + self.eigenvector[1][1] - self.eigenvector[2][2])/2)
        qz = signz * abs(np.sqrt(1 - self.eigenvector[0][0] - self.eigenvector[1][1] + self.eigenvector[2][2])/2)
        '''
        
        #qnew = [qw, qx, qy, qz]
        #print(qnew)
        
        bpy.data.objects[lattice_name].select_set(True)
        bpy.data.objects[lattice_name].rotation_mode = 'QUATERNION'
        bpy.data.objects[lattice_name].rotation_quaternion = quat_z
        
        bpy.ops.transform.rotate(value=math.acos(quat_y[0])*2, orient_axis='Z', orient_type='LOCAL', orient_matrix=((0, 0, 0), (0, 0, 0), (0, 0, 0)), orient_matrix_type='VIEW', mirror=True, proportional='DISABLED', proportional_edit_falloff='SMOOTH', proportional_size=1)
        #bpy.ops.transform.rotate(value=math.acos(quat_z[0])*2, orient_axis='Y', orient_type='LOCAL', orient_matrix=((0, 0, 0), (0, 0, 0), (0, 0, 0)), orient_matrix_type='VIEW', mirror=True, proportional='DISABLED', proportional_edit_falloff='SMOOTH', proportional_size=1)
        
        #bpy.data.objects[lattice_name].rotation_quaternion = quat_y
        bpy.data.objects[lattice_name].location = [self.xmean,self.ymean,self.zmean]
        bpy.data.objects[lattice_name].scale = [2*max(abs(self.pcz)),2*max((self.pcy)),2*max((self.pcx))]
        
    def apply_modifier():
        pass

PCA(obj3).create_lattice()
#===============================    