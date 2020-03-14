import tensorflow as tf
import tflearn as tfl
import os
import sys

layers = ['down_conv_2d', 'conv_2d', 'flatten', 'fully_connected', 'expand', 'up_conv_2d', 'input_layer']
layer_types = [[int, int, str], [int, int, str], [], [int, str], [int, int], [int, int, int, int, str], [int, int, int]]
layer_inputs = ['1 int: number of filters / 2 int: filter size / 3 str: activation',
			    '1 int: number of filters / 2 int: filter size / 3 str: activation',
				'No input parameters',
				'1 int: number of units / 2 str: activation',
				'1 int: x dimension / 2 int: y dimension',
				'1 int: number of filters / 2 int: filter size / 3 int: x dimension / 4 int: y dimension / 5 str: activation',
				'1 int: x dimension / 2 int: y dimension / 3 int: number of channels']

def help():
	h = dict(zip(layers, layer_inputs))

	print
	print 20 * '-' + 'LAYER PARAMETERS FOR CONSTRUCTION' + 20 * '-'
	for key in h.keys():
		s = key.ljust(20) + " --    " + h[key]
		print key.ljust(20) + " --    " + h[key] 
	print 73 * '-'
	print

def str_to_layer_list(s, delimiter = ' '):
	s = s.rsplit('\n')[0]
	params = str.split(s, delimiter)

	print params

	bn = False
	if params[-1] == "batch_norm":
		bn = True
		del params[-1]

	if not params[0] in layers:
		print params[0], ": There is no such layer"
		return None
	if not len(params) - 1 == len(layer_types[layers.index(params[0])]):
		print params[0], "takes", len(layer_types[layers.index(params[0])]), "arguments and got", (len(params) - 1)
		return None

	casted_params = [layer_types[layers.index(params[0])][i](params[i + 1]) for i in xrange(len(params) - 1)]
	if bn:
		casted_params.append("batch_norm")
	casted_params.insert(0, params[0])
	return casted_params


def add_layer(net, lay):
	if lay[0] == 'down_conv_2d':
		net = tfl.conv_2d(net, nb_filter = lay[1], strides = [1,2,2,1], filter_size = lay[2], activation = lay[3])
	elif lay[0] == 'conv_2d':
		net = tfl.conv_2d(net, nb_filter = lay[1], filter_size = lay[2], activation = lay[3])
	elif lay[0] == 'flatten':
		s = net.get_shape()
		net = tf.reshape(net, [tf.shape(net)[0], s[1] * s[2] * s[3]])
	elif lay[0] == 'fully_connected':
		net = tfl.fully_connected(net, n_units = lay[1], activation = lay[2])
	elif lay[0] == 'expand':
		net = tf.reshape(net, [tf.shape(net)[0], lay[1], lay[2], net.get_shape()[1] / lay[1] / lay[2]])
	elif lay[0] == 'up_conv_2d':
		net = tfl.conv_2d_transpose(net, nb_filter = lay[1], filter_size = lay[2], strides = [1,2,2,1]
									  , output_shape = [lay[3], lay[4]]
									  , activation=lay[5], padding="same")
	elif lay[0] == 'input_layer':
		image_prep = tfl.ImagePreprocessing()
		image_prep.add_featurewise_stdnorm(per_channel=True, std = 0.24051991589344662)
		image_prep.add_featurewise_zero_center(per_channel=True, mean = 0.14699117337640238)
		
		net = tfl.layers.input_data(shape=[None, lay[1], lay[2]], data_preprocessing = image_prep)
		net = tf.expand_dims(net, axis = -1)

	if lay[-1] == 'batch_norm':
		net = tfl.batch_normalization(net)

	return net

def read_network(net = None, path = None):
	def _print():
		offset = ''

		last = inp[0][1][1]
		for l in inp:
			if l[1][1] < last:
				offset += '\t'
			if l[1][1] > last:
				offset = offset[:-1]
			last = l[1][1]
			print offset + l[0][0] + ": " + str(l[1])

	inp = []

	read = raw_input('> ')
	while not read.startswith('end'):
		if read == 'help':
			help()
		elif read == 'print':
			_print()
		else:
			as_list = str_to_layer_list(read)
			if not as_list is None:
				net = add_layer(net, as_list)
				inp.append([as_list, net.get_shape().as_list()])

		read = raw_input('> ')

	while read != 'y' and read != 'n' and path is None:
		read = raw_input('Do you want to save the net to a file? (y/n) ')

	if read == 'y' or not path is None:
		if path is None:
			path = raw_input('State path: ')
		f = open(path + "net.nf", 'w')
		for l in inp:
			f.write(str(l[0]).replace("'", '').replace(",", '')[1:-1] + '\n')
		f.write('end')
		f.close()

	return net

def read_network_file(nf, net = None):
	nf = open(nf, 'r')

	buf = nf.readline()
	while not buf.startswith('end'):
		lay = str_to_layer_list(buf)
		net = add_layer(net, lay)

		print net.get_shape()
		print

		buf = nf.readline()

	return net