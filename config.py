from os.path import join
from os.path import realpath

#general
num_models = 10
eps = 0.001

#rendering metadata
img_dim_x = 128
img_dim_y = 128
ortho_scale = 1.0
upscale_factor = 0

#sampling metadata
sampling_method = 'b' #in {'b', 'h'} for sampling from unit ball or hemisphere respectively
sampling_dim = 8

#rasterizing metadata
rasterizing_dim = 512
downscale_factor = 2
grid_dim = rasterizing_dim / downscale_factor

#paths

repository_path = join("/", *realpath(__file__).split("/")[0:-1])
shapenet_origin = join(repository_path, "ShapeNetCore.V2")
shapenet_path = join(repository_path, "shapenet_simple/images/")
shapenet_voxels = join(repository_path, "shapenet_simple/voxels/")
source_code_path = join(repository_path, "python_source_code/")
metadata_path = join(source_code_path, "metadata/")
helpers_path = join(source_code_path, "helpers/")

def model_path(model_name):
    return join(shapenet_path, model_name + "/")
