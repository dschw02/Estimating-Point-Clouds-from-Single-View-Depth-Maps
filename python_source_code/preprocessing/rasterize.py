from multiprocessing import Pool
import multiprocessing
import os
import sys

#custom imports
from find_mount import repository_path
sys.path.append(repository_path)
import config

sys.path.append(os.path.join(repository_path, "python_source_code/helpers"))
from voxel_grid import voxel_grid
#--------------

if not os.path.isdir(config.shapenet_path):
    os.mkdir(config.shapenet_path)


files = [config.shapenet_path + "/" + "%05d" % (n,) + "/models/model_normalized.obj" for n in xrange(config.num_models)]

def process(path):
    path_obj, path_binvox = path, path.split(".")[0]

    grid  = voxel_grid(path_obj, dim = config.rasterizing_dim, eps = config.eps)
    grid.downscale_grid(config.downscale_factor)
    grid.save(path_binvox)

    current = multiprocessing.current_process()
    print current.name + ": " + path_obj + " done"


pool = Pool()
pool.map(process, files)
