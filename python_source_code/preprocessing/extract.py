#!/usr/bin/python

import sys
import os
import glob
import shutil

#custom imports
from find_mount import repository_path
sys.path.append(repository_path)
import config
#--------------


files = filter(lambda f: os.path.isdir(f), glob.glob(os.path.join(config.shapenet_origin, "*/*/")))

if not os.path.isdir(config.shapenet_path):
    os.mkdir(config.shapenet_path)


for i in xrange(len(files)):
    sys.stdout.write("\rextraction of models: %d%%" % int((float(i) / (min(config.num_models, len(files))-1)) * 100))
    sys.stdout.flush()
    shutil.copytree(files[i], os.path.join(config.shapenet_path, "/" + str(i).zfill(5)))

    if i > config.num_models - 2:
	print 
	break



