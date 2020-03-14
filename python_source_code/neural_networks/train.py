import sys
import argparse
import os
import random
import numpy as np
import tensorflow as tf
import tflearn as tfl
from timeit import default_timer as timer
from PIL import Image
import net_constructor
import open3d as o3d

#custom imports
from find_mount import repository_path
sys.path.append(repository_path)
import config

sys.path.append(os.path.join(repository_path, "python_source_code/helpers"))
from voxel_grid import voxel_grid

import net_config_loader
import preloader
import graph_utility
import graph_templates
import prediction_viewer
#--------------

print(tf.__version__)

parser = argparse.ArgumentParser(description="")
parser.add_argument('-p', nargs = '?', help = 'path to net_config')
parser.add_argument('-l', nargs = '?', help = 'name of the model to be loaded')
parser.add_argument('-s', nargs = '?', help = 'name of the model to be saved')

parser.add_argument('-vox', nargs = '?', help = 'name of the model to be saved')
parser.add_argument('-load', nargs = '?', const = 0, help = 'if set - loading, else - training')
parser.add_argument('-train', nargs = '?', const = 0, help = 'if set - training, pretrain for pretraining, train for finalizing')
parser.add_argument('-debug', nargs = '?', const = 0, help = 'if set - feeding data directly to sg_pr')
parser.add_argument('-eval', nargs = '?', const = 0, help = 'if set - evaluating the models output, if "3d" - showing 3d models')
parser.add_argument('-mode', nargs = '?', const = 0, help = 'if set - evaluating the models output, if "3d" - showing 3d models')
parser.add_argument('-c', nargs = '?', const = 0, help = 'if set - evaluating the models output, if "3d" - showing 3d models')


args = parser.parse_args()

net_config = net_config_loader.load_config(args.p, verbose = True)

graph_util = graph_utility.graph_utility(net_config, 105, verbose = True, class_choice = args.c)

graph_templ = graph_templates.graph_templates(net_config, graph_util.mat_sg, graph_util.mat_pr)

#tfl.config.init_graph(gpu_memory_fraction = 0.8)
X, Y1, Y2 = graph_util.X, graph_util.Y1, graph_util.Y2
X_val, Y1_val, Y2_val = graph_util.X_val, graph_util.Y1_val, graph_util.Y2_val
mat_sg, mat_pr = graph_util.mat_sg, graph_util.mat_pr

net_path = os.path.join(args.p, 'net.nf')
log_path = os.path.join(args.p, 'tflearn_logs')				



print
print "----------------------------------------------------------------"
print "------------------------net construction------------------------"

net = net_constructor.read_network_file(net_path)

if args.mode == "pretrain_do":
	net = tf.transpose(net, [0,3,1,2])
	model = graph_templ.construct_model(net, learning_rate = net_config['learning_rate'], loss = 'mean_square', tensorboard = log_path)
elif args.mode == "pretrain":
	depth, mask = tf.split(net, 2, axis = -1)
	depth, mask = tf.transpose(depth, [0,3,1,2]), tf.transpose(mask, [0,3,1,2])
	net = tf.stack([depth, mask], axis = -1)
	model = graph_templ.construct_model(net, learning_rate = net_config['learning_rate'], loss = 'mean_square', tensorboard = log_path)
elif args.mode == "finalize":
	depth, mask = tf.split(net, 2, axis = -1)
	depth, mask = tf.transpose(depth, [0,3,1,2]), tf.transpose(mask, [0,3,1,2])
	net = tf.stack([depth, mask], axis = -1)
	model = graph_templ.training_model_plain(net, learning_rate = net_config['learning_rate_finalize'], loss = graph_templ.custom_sg_pr_loss, tensorboard = log_path)	 
elif args.mode == "finalize_do":
	model = graph_templ.training_model_plain_do(net, learning_rate = net_config['learning_rate_finalize'], loss = 'mean_square', tensorboard = log_path)	
else:
	print "--Mode has to be in ['train', 'finalize', 'finalize_do']--"



if not args.l is None:
	if not args.train is None:
		model.load(os.path.join(args.p, args.l), weights_only = True)
		print "-----Weights loaded from given File------"
		print os.path.join(args.p, args.l)
	else:
		model.load(os.path.join(args.p, args.l))
		print "------Model loaded from given File-------"
		print os.path.join(args.p, args.l)

print "--------------------------net finished--------------------------"
print "----------------------------------------------------------------"
print

if not args.train is None:
	batch_size, val_set, epochs, Y = net_config['batch_size'], net_config["validation_training"], net_config['epochs'], Y1
	if args.mode == "finalize" or args.mode == "finalize_do":
		batch_size, Y, epochs = net_config['batch_size_finalize'], Y2, net_config['epochs_finalize']

	model.fit(X, Y, n_epoch = epochs, validation_set = val_set, batch_size = batch_size)

if not args.s is None:
	model.save(os.path.join(args.p, args.s))

if not args.eval is None:
	if args.l is None:
		print "--Need to pass model to be evaluated--"

	batch_size = 2
	sess = tf.Session()
	dest = os.path.join(args.p, "eval_" + args.l + "/")

	if not os.path.isdir(dest):
		os.mkdir(dest)


	for i in xrange((len(X_val) - 1) / batch_size + 1):
		batch_size = min(batch_size, len(X_val) - i * batch_size)
		
		pred = model.predict(X_val[batch_size * i:batch_size * (i+1)])

		pred = np.transpose(pred, [4,0,1,2,3])

		#viewer = prediction_viewer.prediction_viewer(X_val[batch_size * i:batch_size * (i+1)], pred[:batch_size], pred[:batch_size], Y1_val[batch_size * i:batch_size * (i+1)])
		if args.mode == "pretrain":
			viewer = prediction_viewer.prediction_viewer(X_val[batch_size * i:batch_size * (i+1)], pred[0][:batch_size], pred[1][:batch_size], Y1_val[batch_size * i:batch_size * (i+1)])
		if args.mode == "finalize":
			viewer = prediction_viewer.prediction_viewer(X_val[batch_size * i:batch_size * (i+1)], pred[0][:batch_size], pred[1][:batch_size], Y2_val[batch_size * i:batch_size * (i+1)])

		for j in xrange(batch_size):
			viewer.view_all(2,net_config["base_num_fltrs"] / 2,j, save = True, save_dir = dest,save_name=str(i * batch_size + j))



		if not args.vox is None:
			if args.mode == "pretrain":
				for j in xrange(batch_size):
					grid = voxel_grid(None, scale = False)
					grid.project_to_3d(pred[0][j], mat_sg.eval(session=sess))
					#grid.voxelize(threshold = 1)
					grid.to_mesh(save = dest + str(i * batch_size + j) + ".ply", show = False)
			if args.mode == "finalize":
				mat, _ = graph_util.create_cams(graph_util.sel2, graph_util.sel2, net_config["downscale_factor"])

				for j in xrange(batch_size):
					grid = voxel_grid(None, scale = False)

					mask_bool = (pred[1][j]) > 0
					mask_bool = mask_bool.astype(float)

					pred[0][j] = pred[0][j] * mask_bool

					if not np.count_nonzero(pred[0][j]) == 0:
						grid.project_to_3d(pred[0][j], mat)
						#grid.voxelize(threshold = 10, dim = 16)
						grid.to_mesh(save = dest + str(i * batch_size + j) + ".ply", show = False)


if not args.debug is None:
	batch_size = 4
	data = tf.constant(Y1[:batch_size], dtype = "float32")
	depth, mask = tf.split(data, 2, axis = -1)

	init_op = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init_op)

	#pooled_depth, pooled_mask, _ = graph_templ.sg_pr(depth, mask, mat_sg, mat_pr, show = True, session = sess)
	depth, mask = graph_templ.custom_sg_pr(depth, mask, mat_sg, mat_pr, show = True, session = sess)

	w = depth.eval(session = sess)
	v = mask.eval(session = sess)

	print np.shape(w)

	img = Image.fromarray(w[0,0] * 255.)
	img = img.rotate(90)
	img.show()

	img = Image.fromarray(v[0,0] * 255.)
	img = img.rotate(90)
	img.show()

	#img = grid.render(mat_pr.eval(session=sess)[0], 128, 128)
	#img = Image.fromarray(img * 255.)
	#img = img.rotate(90)
	#img.show()


	#viewer = prediction_viewer.prediction_viewer(X[:batch_size], pooled_depth[:batch_size], pooled_mask[:batch_size], Y2[:batch_size])

	#for i in xrange(8):
	#	viewer.view_all(1,net_config["base_num_fltrs"],i)
