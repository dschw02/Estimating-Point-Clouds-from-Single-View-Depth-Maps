import numpy as np
import tensorflow as tf
import tflearn as tfl
import sys
import os
from PIL import Image

#custom imports
from find_mount import repository_path
sys.path.append(repository_path)
import config

sys.path.append(os.path.join(repository_path, "python_source_code/helpers"))
from voxel_grid import voxel_grid
#--------------

class graph_templates:

	def __init__(self, nc, mat_sg, mat_pr):
		self.net_config = nc
		self.mat_sg = mat_sg
		self.mat_pr = mat_pr

	def pretraining_model_plain(self, image_prep, depth = [3,3], mult = [8,8], latent_layer_size = 512, activation = 'leaky_relu', out_activation = 'sigmoid', loss = 'mean_square', log_dir = '/tmp/tflearn_logs/', net_only = False):
		nc = self.net_config

		input_layer = self.image_input_layer([None, config.img_dim_x / nc["downscale_factor"], config.img_dim_y / nc["downscale_factor"]], image_prep)

		latent_layer = self.depth_encoder(input_layer, activation = activation, depth = depth[0], mult = mult[0], latent_layer = latent_layer_size)
		depth_layer = self.depth_decoder(latent_layer, activation = activation, out_activation = out_activation, depth = depth[1], mult = mult[1], latent_layer = latent_layer_size)

		upscaled_depth, upscaled_mask = tf.split(depth_layer, 2, axis = -1)
		upscaled_depth, upscaled_mask = tf.expand_dims(upscaled_depth, -1), tf.expand_dims(upscaled_mask, -1)

		net = tf.concat([upscaled_depth, upscaled_mask], axis=-1)
		net = tf.transpose(net, [0,3,1,2,4])

		if net_only:
			return net

		return self.construct_model(net, learning_rate = nc["learning_rate"], loss = loss, tensorboard = log_dir)

	def training_model_plain(self, incoming_net, loss = 'mean_square', learning_rate = 0.001,  tensorboard = '/tmp/tflearn_logs/'):
		 depth, mask = tf.split(incoming_net, 2, axis = -1)
		 depth, mask = self.custom_sg_pr(depth, mask, self.mat_sg, self.mat_pr)

		 depth = tf.expand_dims(depth, axis = -1)
		 mask = tf.expand_dims(mask, axis = -1)
		 net = tf.concat([depth, mask], axis = -1)

		 return self.construct_model(net, loss = loss, tensorboard = tensorboard)

	def image_input_layer(self, in_shape, image_prep):
		net = tfl.layers.input_data(shape=in_shape, data_preprocessing=image_prep)
		if len(in_shape) == 3:
			net = tf.expand_dims(net, axis = -1)
		return net

	def construct_model(self, net, verbosity = 0, learning_rate = 0.001, loss = None, metric = "accuracy", tensorboard = "/tmp/tflearn_logs/"):
		adam = tf.train.AdamOptimizer(learning_rate)
		net = tfl.regression(net, optimizer = adam, loss = loss, learning_rate = learning_rate, metric = metric)
		model = tfl.DNN(network=net, tensorboard_verbose=verbosity, tensorboard_dir = tensorboard)

		return model

	def construct_net(self, input_layer, activation = "leaky_relu", out_activation = "sigmoid", depth = 3, mult = 8, latent_layer_size = 512):
		latent_layer = self.depth_encoder(input_layer, activation = activation, depth = depth, mult = mult, latent_layer = latent_layer_size)
		depth_layer = self.depth_decoder(latent_layer, activation = activation, out_activation = out_activation, depth = depth, mult = mult, latent_layer = latent_layer_size)

		upscaled_depth, upscaled_mask = tf.split(depth_layer, 2, axis = -1)
		upscaled_depth, upscaled_mask = tf.expand_dims(upscaled_depth, -1), tf.expand_dims(upscaled_mask, -1)

		net = tf.concat([upscaled_depth, upscaled_mask], axis=-1)
		net = tf.transpose(net, [0,3,1,2,4])

		return net

	def pretraining_model_plain_do(self, image_prep, depth = [3,3], mult = [8,8], latent_layer_size = 512, activation = 'leaky_relu', out_activation = 'sigmoid', loss = 'mean_square', log_dir = '/tmp/tflearn_logs/', net_only = False):
		nc = self.net_config

		input_layer = self.image_input_layer([None, config.img_dim_x / nc["downscale_factor"], config.img_dim_y / nc["downscale_factor"]], image_prep)

		latent_layer = self.depth_encoder(input_layer, activation = activation, depth = depth[0], mult = mult[0], latent_layer = latent_layer_size)
		net = self.depth_decoder_do(latent_layer, activation = activation, out_activation = out_activation, depth = depth[1], mult = mult[1], latent_layer = latent_layer_size)

		net = tf.transpose(net, [0,3,1,2])

		if net_only:
			return net

		return self.construct_model(net, learning_rate = nc["learning_rate"], loss = loss, tensorboard = log_dir)

	def training_model_plain_do(self, net, learning_rate = 0.0001, loss = 'mean_square', tensorboard = '/tmp/tflearn_logs/'):
		depth = self.custom_sg_pr_do(net, self.mat_sg, self.mat_pr)

		return self.construct_model(depth, learning_rate = learning_rate, loss = loss, tensorboard = tensorboard)

	def downscale(self, net, num_filters = 16, filter_size = 2, trained = True, activation = "relu", batch_norm = True):
		if trained:
			net = tfl.conv_2d(net, nb_filter=num_filters, strides=[1,2,2,1], filter_size=filter_size, activation=activation)
			if batch_norm:
				net = tfl.batch_normalization(net)
		else:
			net = tfl.layers.conv.avg_pool_2d(net, filter_size)

		return net

	def upscale(self, net, num_filters, filter_size = 2, trained = True, activation = "relu", batch_norm = True):
		x, y = int(net.get_shape()[-3]), int(net.get_shape()[-2])
		if trained:
			net = tfl.conv_2d_transpose(net, nb_filter=num_filters, filter_size=filter_size, strides = [1,2,2,1]
										, output_shape= [filter_size * x, filter_size * y], activation=activation, padding="same")
			if batch_norm:
				net = tfl.batch_normalization(net)
		else:
			net = tfl.layers.conv.upsample_2d(net, 2)
	
		return net

	def depth_encoder(self, x, mult, depth, latent_layer, activation = "relu", trained_scaling = True, internal_convolutions = 0, internal_fc = 2):
		batch_size, num_fltrs, dim_x, dim_y = self.net_config["batch_size"], self.net_config["base_num_fltrs"] \
			, config.img_dim_x / 2 ** self.net_config["downscale_factor"], config.img_dim_y / 2 ** self.net_config["downscale_factor"]
		
		for i in xrange(depth):
			for j in xrange(internal_convolutions):
				x = tfl.conv_2d(x, nb_filter=num_fltrs * 2 ** (i - 1) * mult, filter_size=(3, 3), activation=activation)
				x = tfl.batch_normalization(x)

			if trained_scaling:
				x = tfl.conv_2d(x, nb_filter=num_fltrs * 2 ** (i - 1) * mult, strides=[1,2,2,1], filter_size=(3, 3), activation=activation)
			else:
				x = tfl.layers.conv.avg_pool_2d(x, 2)	

			x = tfl.batch_normalization(x)

		s = x.get_shape()

		x = tf.reshape(x, [tf.shape(x)[0], s[1] * s[2] * s[3]])

		for i in xrange(internal_fc, 0, -1):
			x = tfl.fully_connected(x, n_units = latent_layer * 2 ** i, activation=activation)
			x = tfl.batch_normalization(x)

		latent = tfl.fully_connected(x, n_units= latent_layer, activation=activation)
	
		return latent

	def depth_decoder(self, latent, mult = 1, depth = 3, latent_layer = 512, activation = "relu", out_activation = "linear", trained_scaling = True, internal_convolutions = 0, internal_fc = 2):
		batch_size, num_fltrs, dim_x, dim_y = self.net_config["batch_size"], self.net_config["base_num_fltrs"] \
			, config.img_dim_x / self.net_config["downscale_factor"], config.img_dim_y / self.net_config["downscale_factor"]

		x = latent

		for i in xrange(internal_fc):	
			x = tfl.fully_connected(x, n_units= latent_layer * 2 ** i, activation=activation)
			x = tfl.batch_normalization(x)

		fc = (latent_layer * 2 ** (internal_fc - 1)) / (dim_x / 2 ** depth * dim_y / 2 ** depth)
		x = tf.reshape(x, [tf.shape(x)[0], dim_x / 2 ** depth, dim_y / 2 ** depth, fc])

		for i in xrange(depth):
			for j in xrange(internal_convolutions):
				x = tfl.conv_2d(x, nb_filter=num_fltrs * 2 ** (depth - i) * mult, filter_size=(3, 3), activation=activation)
				x = tfl.batch_normalization(x)

			if trained_scaling:
				x = tfl.conv_2d_transpose(x, nb_filter=num_fltrs * 2 ** (depth - i) * mult, filter_size=(3, 3), strides = [1,2,2,1]
									  , output_shape= [(dim_x / 2 ** depth) * 2 ** (i+1), (dim_y / 2 ** depth) * 2 ** (i+1)]
									  , activation=activation, padding="same")
				x = tfl.batch_normalization(x)
			else:
				x = tfl.layers.conv.upsample_2d(x, 2)


		x = tfl.conv_2d(x, nb_filter=num_fltrs * 2, filter_size=(3,3), activation=out_activation)

		return x

	def depth_decoder_do(self, latent, mult = 1, depth = 3, latent_layer = 512, activation = "relu", out_activation = "linear", trained_scaling = True, internal_convolutions = 0, internal_fc = 2):
		batch_size, num_fltrs, dim_x, dim_y = self.net_config["batch_size"], self.net_config["base_num_fltrs"] \
			, config.img_dim_x / self.net_config["downscale_factor"], config.img_dim_y / self.net_config["downscale_factor"]

		x = latent

		for i in xrange(internal_fc):	
			x = tfl.fully_connected(x, n_units= latent_layer * 2 ** i, activation=activation)
			x = tfl.batch_normalization(x)

		fc = (latent_layer * 2 ** (internal_fc - 1)) / (dim_x / 2 ** depth * dim_y / 2 ** depth)
		x = tf.reshape(x, [tf.shape(x)[0], dim_x / 2 ** depth, dim_y / 2 ** depth, fc])

		for i in xrange(depth):
			for j in xrange(internal_convolutions):
				x = tfl.conv_2d(x, nb_filter=num_fltrs * 2 ** (depth - i) * mult, filter_size=(3, 3), activation=activation)
				x = tfl.batch_normalization(x)

			if trained_scaling:
				x = tfl.conv_2d_transpose(x, nb_filter=num_fltrs * 2 ** (depth - i) * mult, filter_size=(3, 3), strides = [1,2,2,1]
									  , output_shape= [(dim_x / 2 ** depth) * 2 ** (i+1), (dim_y / 2 ** depth) * 2 ** (i+1)]
									  , activation=activation, padding="same")
				x = tfl.batch_normalization(x)
			else:
				x = tfl.layers.conv.upsample_2d(x, 2)


		x = tfl.conv_2d(x, nb_filter=num_fltrs, filter_size=(3,3), activation=out_activation)

		return x

	def custom_sg_pr(self, incoming_depth, incoming_mask, in_matrices, out_matrices, show = False, session = None):
		#gathering global values for better readability
		#batch_size = self.net_config["batch_size"]
		batch_size = tf.shape(incoming_depth)[0]
		dim_x, dim_y = config.img_dim_x / self.net_config["downscale_factor"], config.img_dim_y / self.net_config["downscale_factor"]
		num_fltrs = self.net_config["base_num_fltrs"]
		eps = self.net_config["epsilon"]
		U = self.net_config["U"]


		def to_4d_cords(tensor):
			ind_x, ind_y = tf.meshgrid(tf.range(dim_x), tf.range(dim_y), indexing="ij")
			ind_x, ind_y = tf.cast(tf.reshape(ind_x, [-1]), dtype="float32"), tf.cast(tf.reshape(ind_y, [-1]), dtype="float32")

			ind_x_tiled = tf.ones([batch_size * num_fltrs, 1]) * ind_x 
			ind_y_tiled = tf.ones([batch_size * num_fltrs, 1]) * ind_y 
			values = tf.reshape(tensor, [batch_size * num_fltrs, dim_x * dim_y])
			ones = tf.ones_like(values)

			output = tf.stack([ind_x_tiled, ind_y_tiled, values, ones], axis=2)

			return output

		def structure_generator(tensor):
			P_inv = tf.reshape(in_matrices, [-1])
			P_inv_tiled = tf.ones([batch_size, 1]) * P_inv
			P_inv_tiled = tf.reshape(P_inv_tiled, [batch_size * num_fltrs, 4, 4])
			
			grid_3d = tf.matmul(P_inv_tiled, tf.transpose(tensor, [0, 2, 1]))

			output = tf.reshape(tf.transpose(grid_3d, [0, 2, 1]), [batch_size, num_fltrs * dim_x * dim_y, 4])

			return output

		def pseudo_renderer(tensor):
			mask = tf.reshape(incoming_mask, [batch_size, dim_x * dim_y * num_fltrs])
			mask = tf.expand_dims(mask, axis = 1)
			mask = tf.tile(mask, [1, num_fltrs, 1])
			mask = tf.reshape(mask, [-1])

			x, y, z, _ = tf.split(tensor, 4, axis = -1)
			cube_mask = (x > -1) & (x < 1) & (y > -1) & (y < 1) & (z > -1) & (z < 1)
			cube_mask = tf.expand_dims(tf.squeeze(cube_mask, axis = -1), axis = 1)
			cube_mask = tf.tile(cube_mask, [1, num_fltrs, 1])
			cube_mask = tf.reshape(cube_mask, [-1])

			valid_mask = (incoming_depth > 0) & (incoming_depth < 1) 
			valid_mask = tf.reshape(valid_mask, [batch_size, dim_x * dim_y * num_fltrs])
			valid_mask = tf.expand_dims(valid_mask, axis = 1)
			valid_mask = tf.tile(valid_mask, [1, num_fltrs, 1])
			valid_mask = tf.reshape(valid_mask, [-1])

			ind_batch, _ = tf.meshgrid(tf.range(batch_size), tf.range(num_fltrs * dim_x * dim_y), indexing="ij")
			ind_batch = tf.expand_dims(ind_batch, axis = -1)

			P_tiled = tf.expand_dims(out_matrices, axis = 0)
			P_tiled = tf.tile(P_tiled, [batch_size, 1,1,1])

			data = tf.expand_dims(tensor, axis = -1)
			data = tf.tile(data, [1,1,1,num_fltrs])
			data = tf.transpose(data, [0,3,2,1])

			rendered = tf.matmul(P_tiled, data)	
			rendered = tf.transpose(rendered, [0,1,3,2])

			indices = tf.where(tf.cast(tf.ones_like(tf.split(rendered, 3, axis = -1))[0], dtype = tf.bool))
			
			rendered = tf.reshape(rendered, [-1, 3])
			ind_x, ind_y, z_values = tf.split(rendered, 3,  axis = 1)

			ind_batch, ind_view, _, _ = tf.split(indices, 4, axis = 1)
			ind_batch, ind_view = tf.cast(ind_batch, dtype = 'float32'), tf.cast(ind_view, dtype = 'float32')

			out = tf.concat([ind_batch, ind_view, ind_x, ind_y, z_values, tf.expand_dims(mask, axis = -1)], axis = 1)

			out = tf.boolean_mask(out, (mask > 0) & cube_mask & valid_mask)

			return out

		def scatter_and_pool(tensor):
			ind_batch, ind_views, ind_x, ind_y, z_values, mask = tf.unstack(tensor, axis = 1)
			ind_x, ind_y = tf.cast(tf.cast(ind_x, dtype = "int32"), dtype = "float32"), tf.cast(tf.cast(ind_y, dtype = "int32"), dtype = "float32")
			tensor = tf.stack([ind_batch, ind_views, ind_x, ind_y, z_values, mask], axis = 1)

			reordered = tf.gather(tensor, tf.nn.top_k(tensor[:, 3], k=tf.shape(tensor)[0]).indices)
			reordered = tf.concat(reordered, axis = 0)

			reordered = tf.gather(reordered, tf.nn.top_k(reordered[:, 2], k=tf.shape(tensor)[0]).indices)
			reordered = tf.concat(reordered, axis = 0)

			reordered = tf.gather(reordered, tf.nn.top_k(reordered[:, 1], k=tf.shape(tensor)[0]).indices)
			reordered = tf.concat(reordered, axis = 0)

			reordered = tf.gather(reordered, tf.nn.top_k(reordered[:, 0], k=tf.shape(tensor)[0]).indices)
			reordered = tf.reverse(reordered, axis = [0])
			reordered = tf.concat(reordered, axis = 0)

			salt = tf.range(0, U, delta = 1)
			salt = tf.reshape(salt, [-1,1])
			salt = tf.tile(salt, [tf.shape(tensor)[0] / U + 1, 1])
			salt = tf.reshape(salt, [-1])[:tf.shape(tensor)[0]]

			ind_batch, ind_views, ind_x, ind_y, z_values, mask = tf.unstack(reordered, axis = 1)

			ind_x = tf.cast(ind_x, dtype = "int32")
			ind_x = ind_x * U + salt
			ind_x = tf.cast(ind_x, dtype = "float32")

			indices = tf.stack([ind_batch, ind_views, ind_x, ind_y], axis = 0)
			indices = tf.cast(indices, dtype="int32")
			indices = tf.transpose(indices)			

			z_values = tf.reshape(z_values, [-1,1])

			scattered_data = tf.scatter_nd(indices, z_values, [batch_size, num_fltrs, dim_x * U, dim_y, 1])
			scattered_data = tf.reshape(scattered_data, [batch_size * num_fltrs, dim_x * U, dim_y, 1])

			pooled_data = tf.nn.max_pool(scattered_data, [1, U, 1, 1], [1, U, 1, 1], padding="VALID")
			pooled_data = tf.reshape(pooled_data, [batch_size, num_fltrs, dim_x, dim_y])


			mask = tf.reshape(mask, [-1,1])

			scattered_mask = tf.scatter_nd(indices, mask, [batch_size, num_fltrs, dim_x * U, dim_y, 1])
			scattered_mask = tf.reshape(scattered_mask, [batch_size * num_fltrs, dim_x * U, dim_y, 1])

			pooled_mask = tf.nn.avg_pool(scattered_mask, [1, U, 1, 1], [1, U, 1, 1], padding="VALID")
			pooled_mask = tf.reshape(pooled_mask, [batch_size, num_fltrs, dim_x, dim_y])

			return pooled_data, pooled_mask


		depth_4d = to_4d_cords(incoming_depth)

		grid_data_3d = structure_generator(depth_4d)	

		rendered_grid_3d = pseudo_renderer(grid_data_3d)

		output_depth, output_mask = scatter_and_pool(rendered_grid_3d)

		return output_depth, output_mask

	def custom_sg_pr_loss(self, y_pred, y_true):
		depth_pred, mask_pred = tf.split(y_pred, 2, axis = -1)
		depth_true, mask_true = tf.split(y_true, 2, axis = -1)

		loss_mask = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=mask_pred, labels=mask_true))
		loss_depth = tf.reduce_sum(tf.abs(tf.boolean_mask(tf.square(depth_true - depth_pred), mask_pred > 0)))	

		return loss_depth + loss_mask

	def custom_sg_pr_do(self, incoming_depth, in_matrices, out_matrices, show = False, session = None):
		#gathering global values for better readability
		#batch_size = self.net_config["batch_size"]
		batch_size = tf.shape(incoming_depth)[0]
		dim_x, dim_y = config.img_dim_x / self.net_config["downscale_factor"], config.img_dim_y / self.net_config["downscale_factor"]
		num_fltrs = self.net_config["base_num_fltrs"]
		eps = self.net_config["epsilon"]
		U = self.net_config["U"]


		def to_4d_cords(tensor):
			ind_x, ind_y = tf.meshgrid(tf.range(dim_x), tf.range(dim_y), indexing="ij")
			ind_x, ind_y = tf.cast(tf.reshape(ind_x, [-1]), dtype="float32"), tf.cast(tf.reshape(ind_y, [-1]), dtype="float32")

			ind_x_tiled = tf.ones([batch_size * num_fltrs, 1]) * ind_x 
			ind_y_tiled = tf.ones([batch_size * num_fltrs, 1]) * ind_y 
			values = tf.reshape(tensor, [batch_size * num_fltrs, dim_x * dim_y])
			ones = tf.ones_like(values)

			mask = tf.reshape(tensor > eps, [batch_size, dim_x * dim_y * num_fltrs])

			output = tf.stack([ind_x_tiled, ind_y_tiled, values, ones], axis=2)

			return output, mask

		def structure_generator(tensor):	
			P_inv = tf.reshape(in_matrices, [-1])
			P_inv_tiled = tf.ones([batch_size, 1]) * P_inv
			P_inv_tiled = tf.reshape(P_inv_tiled, [batch_size * num_fltrs, 4, 4])
			
			grid_3d = tf.matmul(P_inv_tiled, tf.transpose(tensor, [0, 2, 1]))

			output = tf.reshape(tf.transpose(grid_3d, [0, 2, 1]), [batch_size, num_fltrs * dim_x * dim_y, 4])

			return output

		def pseudo_renderer(tensor, mask):
			print mask.get_shape()
			mask = tf.expand_dims(mask, axis = 1)
			print mask.get_shape()
			mask = tf.tile(mask, [1, num_fltrs, 1])
			print mask.get_shape()
			mask = tf.reshape(mask, [-1])
			print mask.get_shape()


			ind_batch, _ = tf.meshgrid(tf.range(batch_size), tf.range(num_fltrs * dim_x * dim_y), indexing="ij")
			ind_batch = tf.expand_dims(ind_batch, axis = -1)

			P_tiled = tf.expand_dims(out_matrices, axis = 0)
			P_tiled = tf.tile(P_tiled, [batch_size, 1,1,1])

			data = tf.expand_dims(tensor, axis = -1)
			data = tf.tile(data, [1,1,1,num_fltrs])
			data = tf.transpose(data, [0,3,2,1])

			rendered = tf.matmul(P_tiled, data)	
			rendered = tf.transpose(rendered, [0,1,3,2])

			indices = tf.where(tf.cast(tf.ones_like(tf.split(rendered, 3, axis = -1))[0], dtype = tf.bool))
			
			rendered = tf.reshape(rendered, [-1, 3])
			ind_x, ind_y, z_values = tf.split(rendered, 3,  axis = 1)

			ind_batch, ind_view, _, _ = tf.split(indices, 4, axis = 1)
			ind_batch, ind_view = tf.cast(ind_batch, dtype = 'float32'), tf.cast(ind_view, dtype = 'float32')


			out = tf.concat([ind_batch, ind_view, ind_x, ind_y, z_values], axis = 1)

			out = tf.boolean_mask(out, mask)

			return out

		def scatter_and_pool(tensor):
			print tensor.get_shape()
			ind_batch, ind_views, ind_x, ind_y, z_values = tf.unstack(tensor, axis = 1)
			ind_x, ind_y = tf.cast(tf.cast(ind_x, dtype = "int32"), dtype = "float32"), tf.cast(tf.cast(ind_y, dtype = "int32"), dtype = "float32")
			tensor = tf.stack([ind_batch, ind_views, ind_x, ind_y, z_values], axis = 1)

			reordered = tf.gather(tensor, tf.nn.top_k(tensor[:, 3], k=tf.shape(tensor)[0]).indices)
			reordered = tf.concat(reordered, axis = 0)

			reordered = tf.gather(reordered, tf.nn.top_k(reordered[:, 2], k=tf.shape(tensor)[0]).indices)
			reordered = tf.concat(reordered, axis = 0)

			reordered = tf.gather(reordered, tf.nn.top_k(reordered[:, 1], k=tf.shape(tensor)[0]).indices)
			reordered = tf.concat(reordered, axis = 0)

			reordered = tf.gather(reordered, tf.nn.top_k(reordered[:, 0], k=tf.shape(tensor)[0]).indices)
			reordered = tf.reverse(reordered, axis = [0])
			reordered = tf.concat(reordered, axis = 0)

			salt = tf.range(0, U, delta = 1)
			salt = tf.reshape(salt, [-1,1])
			salt = tf.tile(salt, [tf.shape(tensor)[0] / U + 1, 1])
			salt = tf.reshape(salt, [-1])[:tf.shape(tensor)[0]]

			ind_batch, ind_views, ind_x, ind_y, z_values = tf.unstack(reordered, axis = 1)

			ind_x = tf.cast(ind_x, dtype = "int32")
			ind_x = ind_x * U + salt
			ind_x = tf.cast(ind_x, dtype = "float32")

			indices = tf.stack([ind_batch, ind_views, ind_x, ind_y], axis = 0)
			indices = tf.cast(indices, dtype="int32")
			indices = tf.transpose(indices)			

			z_values = tf.reshape(z_values, [-1,1])

			scattered_data = tf.scatter_nd(indices, z_values, [batch_size, num_fltrs, dim_x * U, dim_y, 1])
			scattered_data = tf.reshape(scattered_data, [batch_size * num_fltrs, dim_x * U, dim_y, 1])

			pooled = tf.nn.max_pool(scattered_data, [1, U, 1, 1], [1, U, 1, 1], padding="VALID")
			pooled = tf.reshape(pooled, [batch_size, num_fltrs, dim_x, dim_y])

			return pooled


		depth_4d, mask = to_4d_cords(incoming_depth)

		grid_data_3d = structure_generator(depth_4d)	

		rendered_grid_3d = pseudo_renderer(grid_data_3d, mask)

		output_depth = scatter_and_pool(rendered_grid_3d)

		return output_depth

	def sg_pr(self, incoming_depth, incoming_mask, in_matrices, out_matrices, show=False, session=None):
		#gathering global values for better readability
		#batch_size = self.net_config["batch_size"]
		batch_size = tf.shape(incoming_depth)[0]
		dim_x, dim_y = config.img_dim_x / self.net_config["downscale_factor"], config.img_dim_y / self.net_config["downscale_factor"]
		num_fltrs = self.net_config["base_num_fltrs"]
		eps = self.net_config["epsilon"]
		U = self.net_config["U"]
				
		#--------------helper functions--------------#
	
		# converts images of shape [X, height, width] to shape [X * height * width, 4]
		def to_4d_cords(tensor):
			ind_x, ind_y = tf.meshgrid(tf.range(dim_x), tf.range(dim_y), indexing="ij")
			ind_x, ind_y = tf.cast(tf.reshape(ind_x, [-1]), dtype="float32"), tf.cast(tf.reshape(ind_y, [-1]), dtype="float32")

			ind_x_tiled = tf.ones([batch_size * num_fltrs, 1]) * ind_x 
			ind_y_tiled = tf.ones([batch_size * num_fltrs, 1]) * ind_y 
			values = tf.reshape(tensor, [batch_size * num_fltrs, dim_x * dim_y])
			ones = tf.ones_like(values)

			cutoff_mask = values > eps 
			cutoff_mask = tf.reshape(cutoff_mask, [batch_size, 1, dim_x * dim_y * num_fltrs, 1])
			cutoff_mask = tf.tile(cutoff_mask, [1, num_fltrs, 1, 1])

			output = tf.stack([ind_x_tiled, ind_y_tiled, values, ones], axis=2)
		
			return output, cutoff_mask

		# projects points in image-space to 3d-space by using the inverse projection matrix
		def structure_generator(tensor):	
			P_inv = tf.reshape(in_matrices, [-1])
			P_inv_tiled = tf.ones([batch_size, 1]) * P_inv
			P_inv_tiled = tf.reshape(P_inv_tiled, [batch_size * num_fltrs, 4, 4])
			
			grid_3d = tf.matmul(P_inv_tiled, tf.transpose(tensor, [0, 2, 1]))

			output = tf.reshape(tf.transpose(grid_3d, [0, 2, 1]), [batch_size, num_fltrs * dim_x * dim_y, 4])

			return output

		# renders all given points
		def pseudo_renderer(tensor):
			P = tf.reshape(out_matrices, [-1])
			P_tiled = tf.ones([batch_size, 1]) * P
			P_tiled = tf.reshape(P_tiled, [batch_size, num_fltrs, 3, 4])

			tensor = tf.reshape(tensor, [batch_size, 1, dim_x * dim_y * num_fltrs, 4])
			tensor = tf.tile(tensor, [1, num_fltrs, 1, 1])

			x, y, z, _ = tf.split(tensor, 4, axis=3)
			thr = 1.
			cube_mask = tf.reshape((x > -thr) & (y > -thr) & (z > -thr) & (x <= thr) & (y <= thr) & (z <= thr)
						, [batch_size, num_fltrs, dim_x * dim_y * num_fltrs, 1])

			grid_3d = tf.matmul(P_tiled, tf.transpose(tensor, [0, 1, 3, 2]))
	
			output = tf.transpose(grid_3d, [0, 1, 3, 2])

			return output, cube_mask

		def cut_outliers(tensor, mask, cube_mask, cutoff_mask):
			mask = tf.tile(tf.reshape(mask, [batch_size, 1, num_fltrs, dim_x, dim_x]), [1, num_fltrs, 1, 1, 1])
			x, y, z = tf.split(tensor, 3, axis=3)

			ind_x_upscaled = tf.cast(x * U, dtype="int32")
			ind_y_upscaled = tf.cast(y * U, dtype="int32")

			batch_index, viewpoint_index, _ = tf.meshgrid(tf.range(batch_size), tf.range(num_fltrs), tf.range(num_fltrs * dim_x * dim_y), indexing="ij")

			ind_x_upscaled_flat = tf.reshape(ind_x_upscaled, [-1])
			ind_y_upscaled_flat = tf.reshape(ind_y_upscaled, [-1])
			batch_index_flat = tf.reshape(batch_index, [-1])
			viewpoint_index_flat = tf.reshape(viewpoint_index, [-1])

			mask_flat = tf.reshape(mask, [-1])
			values_flat = tf.reshape(z, [-1])

			int_tensor_all = tf.stack([batch_index_flat, viewpoint_index_flat, ind_x_upscaled_flat, ind_y_upscaled_flat], axis=1)
			float_tensor_all = tf.stack([mask_flat, values_flat], axis=1)

			mask_valid = (x >= 0) & (x < dim_x) & (y >= 0) & (y < dim_y) & cube_mask & cutoff_mask
			mask_valid_flat = tf.reshape(mask_valid, [-1])

			return tf.boolean_mask(int_tensor_all, mask_valid_flat), tf.boolean_mask(float_tensor_all, mask_valid_flat)

		def scatter_and_pool_values(int_tensor_true, float_tensor_true):
			indices = int_tensor_true

			mask_values, z_values = tf.unstack(float_tensor_true, axis=1)
			values = tf.stack([z_values, mask_values, tf.ones_like(z_values)], axis=1)

			print
			print "scatter reference"
			print "indices: ", indices.get_shape()
			print "values: ", values.get_shape()
			print "shape: ", [batch_size, num_fltrs, dim_x * U, dim_y * U, 3]
			print

			scattered_data = tf.scatter_nd(indices, values, [batch_size, num_fltrs, dim_x * U, dim_y * U, 3])
			scattered_data = tf.reshape(scattered_data, [batch_size * num_fltrs, dim_x * U, dim_y * U, 3])

			pooled = tf.nn.max_pool(scattered_data, [1, U, U, 1], [1, U, U, 1], padding="VALID")
			pooled = tf.reshape(pooled, [batch_size, num_fltrs, dim_x, dim_y, 3])

			return pooled

		def scatter_and_pool_mask(int_tensor_false, float_tensor_false):
			indices = int_tensor_false
			values = tf.reshape(tf.unstack(float_tensor_false, axis=1)[0], [-1, 1])

			scattered_data = tf.scatter_nd(indices, values, [batch_size, num_fltrs, dim_x * U, dim_y * U, 1])
			
			scattered_data = tf.reshape(scattered_data, [batch_size * num_fltrs, dim_x * U, dim_y * U, 1])

			pooled = tf.nn.avg_pool(scattered_data, [1, U, U, 1], [1, U, U, 1], padding="VALID")
			pooled = tf.reshape(pooled, [batch_size, num_fltrs, dim_x, dim_y])

			return pooled

		#-----------------main code------------------#

		depth_4d, cutoff_mask = to_4d_cords(incoming_depth)

		grid_data_3d = structure_generator(depth_4d)	

		rendered_grid_3d, cube_mask = pseudo_renderer(grid_data_3d)

		int_tensor_valid, float_tensor_valid = cut_outliers(rendered_grid_3d, incoming_mask, cube_mask, cutoff_mask)

		depth_mask = tf.unstack(float_tensor_valid, axis=1)[1] > 0

		int_tensor_true, float_tensor_true = tf.boolean_mask(int_tensor_valid, depth_mask) \
											, tf.boolean_mask(float_tensor_valid, depth_mask)

		int_tensor_false, float_tensor_false = tf.boolean_mask(int_tensor_valid, ~depth_mask) \
											, tf.boolean_mask(float_tensor_valid, ~depth_mask)

		pooled_values = scatter_and_pool_values(int_tensor_true, float_tensor_true)
		pooled_false_mask = scatter_and_pool_mask(int_tensor_false, float_tensor_false)
		pooled_depth, pooled_true_mask, pooled_collision = tf.unstack(pooled_values, 3, axis=4)

		final_mask = tf.where(pooled_true_mask > 0, pooled_true_mask \
					, tf.where(pooled_false_mask < 0, pooled_false_mask, tf.ones_like(pooled_depth) * -1.0))

		
		if show:
			grid_flat = grid_data_3d[0]
			mask_valid = cube_mask & cutoff_mask
			mask_valid = tf.split(mask_valid, self.net_config["base_num_fltrs"], axis = 1)[0]
			mask_valid = tf.squeeze(mask_valid)
			mask_valid_flat = tf.reshape(mask_valid[0], [-1])

			data = tf.boolean_mask(grid_flat, mask_valid_flat)
			data = tf.transpose(data)
			data = data.eval(session=session)

			grid_data = [data[0], data[1], data[2]]
			grid = voxel_grid(grid_data, scale = False, viewed_points = 10000)

			return grid	

		return pooled_depth, final_mask, pooled_collision	

	def grid_loss(self, y_pred, y_true):
		batch_size, num_fltrs, dim_x, dim_y = self.net_config['batch_size'], self.net_config['base_num_fltrs'] \
			, config.img_dim_x  / 2 ** self.net_config["downscale_factor"], config.img_dim_y  / 2 ** self.net_config["downscale_factor"]

		depth, mask = tf.split(y_pred, 2, axis = -1)
		data_pred, mask_pred, collision	= self.sg_pr(depth, mask, self.mat_sg, self.mat_pr)

		data_true, mask_true			= tf.split(y_true, 2, axis=-1)
		data_true, mask_true			= tf.squeeze(data_true, axis=-1), tf.squeeze(mask_true, axis=-1)

		mask_pred = tf.squeeze(mask_pred)
		mask_true = tf.reshape(mask_true, [batch_size, num_fltrs, dim_x, dim_y])

		loss_mask = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=mask_pred, labels=mask_true))
		loss_depth = tf.reduce_sum(tf.abs(tf.boolean_mask(tf.square(data_true - data_pred), tf.equal(collision, 1))))
		loss_mask, loss_depth = loss_mask / (batch_size * num_fltrs), loss_depth / (batch_size * num_fltrs)

		return loss_depth + loss_mask