import argparse
import os
import numpy
from PIL import Image
import bpy
import bmesh
import sys

#custom imports
from find_mount import repository_path
sys.path.append(repository_path)
import config
#--------------

parser = argparse.ArgumentParser(description="")

parser.add_argument('--background', nargs = '?', help = 'blender uses this')
parser.add_argument('--python', nargs = '?', help = 'blender uses this')
parser.add_argument('-n', nargs = '?', help = 'number of the model')

args = parser.parse_args()

def reset_blend():
    bpy.ops.wm.read_factory_settings()

    for scene in bpy.data.scenes:
        for obj in scene.objects:
            scene.objects.unlink(obj)

    for bpy_data_iter in (
            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.lamps,
            bpy.data.cameras,
    ):
        for id_data in bpy_data_iter:
            bpy_data_iter.remove(id_data)

def set_context_data():
    scn = bpy.context.scene
    scn.world.horizon_color = (1, 1, 1)

    scn.render.resolution_x = config.img_dim_x
    scn.render.resolution_y = config.img_dim_y
    scn.render.resolution_percentage = 100

    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center="BOUNDS")

def create_lamp():
    def random_three_vector():
        """
        Generates a random 3D unit vector (direction) with a uniform spherical distribution
        Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
        :return:
        """
        phi = numpy.random.uniform(0, numpy.pi * 2)
        costheta = numpy.random.uniform(-1, 1)

        theta = numpy.arccos(costheta)
        x = numpy.sin(theta) * numpy.cos(phi)
        y = numpy.sin(theta) * numpy.sin(phi)
        z = numpy.cos(theta)

        return (x * lamp_distance * 2, y * lamp_distance * 2, abs(z) * lamp_distance)
    
    scn = bpy.context.scene
    
    lamp_data = bpy.data.lamps.new(name="New Lamp", type='SUN')
    lamp = bpy.data.objects.new(name="lamp", object_data=lamp_data)
    scn.objects.link(lamp)

    lamp.location = random_three_vector()
    lamp.select = True

def create_camera(cam, target, name):
    scn = bpy.context.scene
    camera = bpy.data.objects.new(name, bpy.data.cameras.new(name))

    camera.data.type = "ORTHO"
    camera.data.ortho_scale = config.ortho_scale

    scn.objects.link(camera)
    camera.location = (cam[0], cam[1], cam[2])
    m = camera.constraints.new('TRACK_TO')
    m.target = target
    m.track_axis = 'TRACK_NEGATIVE_Z'
    m.up_axis = 'UP_Y'

    scn.camera = camera

    return camera

def set_rot_center_to_origin(target):
    scn = bpy.context.scene
    rot_centre = bpy.data.objects.new('rot_centre', None)
    scn.objects.link(rot_centre)
    target.location = [0, 0, 0]
    rot_centre.location = target.location

def render(mode, model_path, output_path, target):
    bpy.context.scene.objects.active = target

    if mode == "BW":
        bpy.data.scenes['Scene'].render.use_textures = False
    else:
        bpy.data.scenes['Scene'].render.use_textures = True

    bpy.data.scenes['Scene'].render.image_settings.file_format = 'PNG'
    bpy.data.scenes['Scene'].render.image_settings.color_mode = mode
    bpy.data.scenes['Scene'].render.filepath = os.path.join(output_path, str(j).zfill(3))
    bpy.ops.render.render(write_still=True)





model_path = os.path.join(os.path.join(config.shapenet_path, "models"), os.path.join(str(args.n), 'models/model_normalized.obj'))
lamp_distance = 3
cams = numpy.load(os.path.join(config.metadata_path, "sample.npy"))

reset_blend()
set_context_data()
create_lamp()

bpy.ops.import_scene.obj(filepath=model_path, split_mode='OFF')

#for m in bpy.data.materials:
#    m.use_shadeless = True

bpy.context.scene.objects.active = bpy.context.selected_objects[0]
obj = bpy.context.selected_objects[0]


if (len(bpy.context.selected_objects) > 1):
    bpy.ops.object.join()

set_rot_center_to_origin(obj)

for j in range(len(cams)):
    print(len(cams))
    camera = create_camera(cams[j], obj, str(j))
    render("RGB", args.n, os.path.join(os.path.join(config.shapenet_path, "rendered_images"), args.n), obj)

reset_blend()


bpy.ops.wm.quit_blender()

