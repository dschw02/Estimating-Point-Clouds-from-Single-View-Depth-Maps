import numpy as np
import tensorflow as tf
import tflearn as tfl
import preloader
import random
import sys
import os
from timeit import default_timer as timer
import numpy

#custom imports
from find_mount import repository_path
sys.path.append(repository_path)
import config

import graph_templates
import preloader
#--------------

class graph_utility:
	def __init__(self, net_config, num_cams, verbose = False, class_choice = None):	
		self.net_config = net_config
		self.num_cams = num_cams
		self.input_view = 22
		
		rand = np.arange(0, num_cams)
		rand = np.delete(rand, self.input_view)	
		random.Random(net_config['seed']).shuffle(rand)
		self.sel1, self.sel2 =  rand[:net_config['base_num_fltrs']], rand[net_config['base_num_fltrs']:net_config['base_num_fltrs'] * 2]
		self.sel1.sort(), self.sel2.sort()

		self.X, self.Y1, self.Y2, self.X_val, self.Y1_val, self.Y2_val = self.get_training_data(class_choice)
		
		self.mat_sg, self.mat_pr = self.get_projection_matrices()

		if self.net_config["downscale_factor"] > 1:
			print "creating new matrices"
			self.mat_sg, self.mat_pr = self.create_cams(self.sel1, self.sel2, net_config["downscale_factor"])

		self.mat_sg = tf.constant(np.array(self.mat_sg), dtype='float32')
		self.mat_pr = tf.constant(np.array(self.mat_pr), dtype='float32')
	
		if verbose:
			self.to_cmd()

	def image_prep(self, stdn = True, mean = True):
		image_prep = tfl.ImagePreprocessing()
		if stdn: image_prep.add_featurewise_stdnorm(per_channel=True)
		if mean: image_prep.add_featurewise_zero_center(per_channel=True)
		return image_prep

	def get_training_data(self, class_choice):
		X  = preloader.preloader(config.shapenet_path, self.input_view, class_choice = class_choice)
		X_val = X.split(self.net_config["validation_set"])
		if self.net_config["downscale_factor"] > 1:
			X = preloader.scaled_preloader(X, self.net_config["downscale_factor"])
			X_val = preloader.scaled_preloader(X_val, self.net_config["downscale_factor"])

		Y1_pre = preloader.preloader(config.shapenet_path, self.sel1, class_choice = class_choice)
		Y1_pre_val = Y1_pre.split(self.net_config["validation_set"])
		if self.net_config["downscale_factor"] > 1:
			Y1_pre = preloader.scaled_preloader(Y1_pre, self.net_config["downscale_factor"])
			Y1_pre_val = preloader.scaled_preloader(Y1_pre_val, self.net_config["downscale_factor"])
		Y1 = preloader.mask_wrapper(Y1_pre, Y1_pre)
		Y1_val = preloader.mask_wrapper(Y1_pre_val, Y1_pre_val)

		Y2_pre = preloader.preloader(config.shapenet_path, self.sel2, class_choice = class_choice)
		Y2_pre_val = Y2_pre.split(self.net_config["validation_set"])
		if self.net_config["downscale_factor"] > 1:
			Y2_pre = preloader.scaled_preloader(Y2_pre, self.net_config["downscale_factor"])
			Y2_pre_val = preloader.scaled_preloader(Y2_pre_val, self.net_config["downscale_factor"])
		Y2 = preloader.mask_wrapper(Y2_pre, Y2_pre)
		Y2_val = preloader.mask_wrapper(Y2_pre_val, Y2_pre_val)

		return X, Y1, Y2, X_val, Y1_val, Y2_val

	def get_projection_matrices(self):
		MAT_SG = np.load(os.path.join(config.metadata_path, "P_.npy"))[self.sel1]
		MAT_PR = np.load(os.path.join(config.metadata_path, "P.npy"))[self.sel2]
		return MAT_SG, MAT_PR

	def to_cmd(self):
		print
		print "----------------------------------------------------------------"
		print "--------------------------graph utility-------------------------"

		print "Structure Generator Views".ljust(25),    " ->      ", self.sel1
		print "Pseudo Renderer Views".ljust(25),        " ->      ", self.sel2

		print "--------------------------graph utility-------------------------"
		print "----------------------------------------------------------------"
		print

	def create_cams(self, sel1, sel2, factor):
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
		    scale_x, scale_y = config.img_dim_x / 2. / factor, config.img_dim_y / 2. / factor
		    x, y = scale_x, scale_y

		    return numpy.array([[scale_x, 0      , 0   	      , scale_x],
				       	 		[0      , scale_y, 0          , scale_y],
				        		[0      , 0      , 1. / 2   , 0      ]])
		    

		def get_full_projection_P(K, R, t):
		    RT = get_rigid_transformation_RT(R, t)   
		    return numpy.matmul(K, RT)

		def get_inverse_projection_P_(K, R, t):
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

		return P_[sel1], P[sel2]

