import net_constructor
import os

config = dict(
	batch_size = 16,
	learning_rate = 0.0001,
	epochs = 1,
	base_num_fltrs = 8,
	U = 10,
	seed = 0,
)

float_keys = ["learning_rate", "learning_rate_finalize", "epsilon", "validation_set", "validation_training"]
int_keys = ["batch_size", "batch_size_finalize", "epochs", "epochs_finalize", "base_num_fltrs", "U", "seed", "downscale_factor"]

def load_config(path, verbose = False):
	if verbose:
		print
		print "----------------------------------------------------------------"	
		print "---------------------------net config---------------------------"

	f = open(path + "net_config")
	
	buf = f.readline()
	while not buf == "" or buf == " ":
		if buf.startswith('#'):
			buf = f.readline()
			continue

		buf = buf[:-1].split(" ")

		if buf[0] in float_keys:
			config[buf[0]] = float(buf[1])
		elif buf[0] in int_keys:
			config[buf[0]] = int(buf[1])
		else:
			config[buf[0]] = buf[1] 	

		if verbose:
			print buf[0].ljust(25), " ->      ", config[buf[0]]

		buf = f.readline()

	config['directory'] = path

	if verbose:
		print "---------------------------net config---------------------------"
		print "----------------------------------------------------------------"
		print

	return config

def save_config(path):
	f = open(path, 'w')

	for k in config.keys():
		f.write(k + ' ' + config[k] + '\n')

	f.close()

def create_net_configuration(path):
	print "---------------------Creating Net-Config---------------------"

	for key in float_keys:
		config[key] = raw_input(key + ' (float): ')

	for key in int_keys:
		config[key] = raw_input(key + ' (int): ')

	save_config(os.path.join(path, 'net_config'))

	print 
	print "------------------------Creating Net-------------------------"

	net_constructor.read_network(path = path)
