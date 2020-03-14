import argparse
import os
import numpy
from PIL import Image
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import sys
import multiprocessing 
import gc
from contextlib import contextmanager

#custom imports
from find_mount import repository_path
sys.path.append(repository_path)
import config

sys.path.append(os.path.join(repository_path, "python_source_code/helpers"))
from voxel_grid import voxel_grid
#--------------

@contextmanager
def pseudo_render(path_to_model):
	model_path = path_to_model
	data_path = os.path.join(path_to_model[:-16], "images/pseudo_rendered/")
	img_path = os.path.join(data_path, path_to_model.split("/")[-1][:-4] + "/")
	
        output_path = model_path[:-16] + "images/"
	
	if os.path.isfile(output_path + path_to_model.split("/")[-1][:-4]):
	    print output_path + path_to_model.split("/")[-1], " already computed"
	    return

	try:
            point_cloud = voxel_grid(model_path, dim = config.grid_dim, upscale_factor = config.upscale_factor)
	except:
	    with open('corrupt_data', 'w') as f:
		f.write(model_path + ' is corrupted\n')	
	    return

	try:
	    P = numpy.load(os.path.join(config.metadata_path,"P.npy"))
	except:
	    with open('corrupt_data', 'w') as f:
		f.write(os.path.join(config.metadata_path,"P.npy") + ' is corrupted\n')	
	    return

	#if not os.path.isdir(data_path):
	#    os.mkdir(data_path)
	if not os.path.isdir(img_path):
	    os.mkdir(img_path)


	float_npy = numpy.zeros((len(P), config.img_dim_x, config.img_dim_y))

	for i in xrange(len(P)):
		cur_img_path = os.path.join(img_path, str(i).zfill(3) + ".png")
		out = point_cloud.render(P[i], config.img_dim_x, config.img_dim_y)
		
		img = Image.fromarray(out * 255.)
		img = img.rotate(90)
		img = img.convert("RGB")

		float_npy[i] = out	    
		img.save(cur_img_path)
	
	numpy.savez_compressed(output_path + path_to_model.split("/")[-1][:-4], float_npy)

	current = multiprocessing.current_process()
	print current.name + ": " + model_path.split("/")[-3] + " done"
	
	return 
     



models = [os.path.join(config.shapenet_path, f) for f in os.listdir(config.shapenet_path)]

models.sort()

pool = multiprocessing.Pool()
pool.map(pseudo_render, models)

