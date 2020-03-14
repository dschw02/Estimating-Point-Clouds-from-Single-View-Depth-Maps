from voxel_grid import voxel_grid
import argparse
import sys
import os
import open3d as o3d
import numpy

#custom imports
from find_mount import repository_path
sys.path.append(repository_path)
import config
#--------------

parser = argparse.ArgumentParser(description="")
parser.add_argument('-n', nargs = 1, type = int, help = 'model_number')
parser.add_argument('-ply', nargs = 1, type = str, help ='path to ply')
args = parser.parse_args()


if not args.n is None:
	model_number = "%05d" % (args.n[0],)
	model_path = config.shapenet_voxels + model_number + ".npz"

	if os.path.isfile(model_path):
		grid = voxel_grid(model_path)
		grid.to_mesh(show = True)

if not args.ply is None:
	vox = o3d.io.read_point_cloud(args.ply[0])
	o3d.visualization.draw_geometries([vox])