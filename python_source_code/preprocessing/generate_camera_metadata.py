#!/usr/bin/python

import numpy
import numpy.linalg
import sys
import argparse
import os

#custom imports
from find_mount import repository_path
sys.path.append(repository_path)
import config
#--------------

def get_rotation_matrix_R(C, p, up_direction):
    u = numpy.array(up_direction)
    C = numpy.array(C)
    p = numpy.array(p)

    L = p - C
    L = L / numpy.linalg.norm(L)
    

    s = numpy.cross(L, u)
    if numpy.linalg.norm(s) != 0:    
	   s = s / numpy.linalg.norm(s)
    else:
	   s = numpy.array([1,0,0])

    u_ = numpy.cross(s, L)

    R = numpy.array([s, u_, -L])

    return R

def get_rigid_transformation_RT(R, t):
    tmp = numpy.transpose(numpy.vstack([numpy.transpose(R),t]))

    return numpy.vstack([tmp, numpy.array([0,0,0,1])])

def get_translation_vector_t(R, C):
    return numpy.dot(R, C)

def get_projection_matrix_K():
    scale_x, scale_y = config.img_dim_x / 2., config.img_dim_y / 2.
    x, y = scale_x, scale_y

    return numpy.array([[scale_x, 0      , 0   	      , scale_x],
		       	 		[0      , scale_y, 0          , scale_y],
		        		[0      , 0      , 1. / 2   , 0      ]])
    

def get_full_projection_P(K, R, t):
    RT = get_rigid_transformation_RT(R, t)
    
    return numpy.matmul(K, RT)

def get_inverse_projection_P_(K, R, t):
	'''
	K44 = numpy.vstack([K, numpy.array([0,0,0,1])])
	K44_ = numpy.linalg.inv(K44)

	R44_ = numpy.linalg.inv(R)
	R44_.resize([4,4])
	R44_[3][3] = 1

	t44_ = numpy.identity(4)
	t44_[0][3], t44_[1][3], t44_[2][3] = -t
	'''

	return numpy.linalg.inv(numpy.vstack([get_full_projection_P(K, R, t), [0,0,0,1]]))
	return numpy.matmul(R44_, numpy.matmul(t44_, K44_))

cams = numpy.load(os.path.join(config.metadata_path, "sample.npy"))

if not os.path.isdir(config.metadata_path):
    os.mkdir(config.metadata_path)

R = numpy.array([get_rotation_matrix_R(cam, (0, 0, 0), (0, 1, 0)) for cam in cams])
t = numpy.array([get_translation_vector_t(R[i], cams[i]) for i in xrange(len(cams))])
K = numpy.array([get_projection_matrix_K() for cam in cams])

P = numpy.array([get_full_projection_P(K[i], R[i], t[i]) for i in xrange(len(cams))])
P_ = numpy.array([get_inverse_projection_P_(K[i], R[i], t[i]) for i in xrange(len(cams))])

numpy.save(os.path.join(config.metadata_path, "P"), P)
numpy.save(os.path.join(config.metadata_path, "P_"), P_)

