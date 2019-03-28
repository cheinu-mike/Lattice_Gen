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
from mathutils import Matrix

#*******SELECT THE OBJECT YOU WANT******
obj = bpy.data.objects['Cube']
obj2 = bpy.data.objects['Cube.001']
obj3 = bpy.data.objects['Cube.002']
selected_obj = bpy.context.selected_objects
#***************************************


#returns list of the selected objects (bpy.context.selected_objects)
def get_objname(selected_objects):
    return [item.name for item in selected_objects]
        
get_objname(bpy.context.selected_objects)

#Extract the GLOBAL vertex coordinates of the object
def get_localvert_coor2(active_object):
    all_verts = []
    for objs in active_object:
        global_verts = np.array(objs.matrix_world.translation)
        if objs.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(objs.data)
            verts = [np.array(vert.co) + global_verts for vert in bm.verts]
            all_verts += verts
        else:
            verts = [np.array(vert.co) + global_verts for vert in objs.data.vertices]
            all_verts += verts
    return all_verts

def get_localvert_coor(active_object):
    all_verts = []
    for objs in active_object:
        global_verts = np.array(objs.matrix_world.translation)
        if objs.mode == 'EDIT':
            bm = bmesh.from_edit_mesh(objs.data)
            verts = [np.array(vert.co) + global_verts for vert in bm.verts if vert.select]
            all_verts += verts
        else:
            verts = [np.array(vert.co) + global_verts for vert in objs.data.vertices]
            all_verts += verts
    return all_verts    

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
        self.identity = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.basex = np.array([1,0,0])
        self.basey = np.array([0,1,0])
        self.basez = np.array([0,0,1])
        self.objname = selected_object[0].name  #CHANGE FOR MULTIOBJECT
        self.localverts = get_localvert_coor(selected_object) #CHANGE FOR MULTIOBJECT AND EDITMODE
        self.globalverts = np.array(self.localverts) #+ np.array(selected_object[0].matrix_world.translation) #CHANGE FOR MULTIOBJECT AND EDITMODE
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
        
        bpy.ops.object.mode_set(mode = 'OBJECT')
        for obj in bpy.data.objects:
            obj.select_set(False)
        
        lattice_name = 'PCA_' + str(self.objname) + '_lattice'
        
        #if any(obj == bpy.data.objects[lattice_name] for obj in bpy.data.objects):
        
        bpy.ops.object.add(type='LATTICE', view_align=False, enter_editmode=False, location=(self.xmean, self.ymean, self.zmean))
        bpy.data.objects['Lattice'].name = lattice_name
        
        scale_matrix = np.array([[2*max((self.pcx)),0,0],[0,2*max((self.pcy)),0],[0,0,2*max((self.pcz))]])
        world_matrix = np.dot(self.eigenvector,scale_matrix)
        
        bpy.data.objects[lattice_name].select_set(True)
        bpy.data.objects[lattice_name].rotation_mode = 'QUATERNION'
        
        bpy.data.objects[lattice_name].matrix_world = Matrix(((self.eigenvector[0][0], self.eigenvector[1][0], -self.eigenvector[2][0], self.xmean),
                                                              (self.eigenvector[0][1], self.eigenvector[1][1], -self.eigenvector[2][1], self.ymean),
                                                              (self.eigenvector[0][2], self.eigenvector[1][2], -self.eigenvector[2][2], self.zmean),
                                                              (0.0, 0.0, 0.0, 1.0)))

        bpy.data.objects[lattice_name].scale = [2*max(abs(self.pcx)),2*max((self.pcy)),2*max((self.pcz))]
        
    def apply_modifier(self):
        pass
if __init__:'__main__':
    PCA(selected_obj).create_lattice()
#===============================    
