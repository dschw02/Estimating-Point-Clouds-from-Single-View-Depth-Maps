#!/usr/bin/python

#script to generate and save uniformally distributed points on a hemisphere or ball
#for usability check help in command line
#"n" is not the number of sampled points, but the dimension of the square.
#actual #points = 2*n^2 - 4*(n - 1) for balls and n^2 for hemispheres 

import numpy
import sys
import argparse
import os

#custom imports
from find_mount import repository_path
sys.path.append(repository_path)
import config
#--------------

def sample_hemisphere(nb_cameras_root=16):
    def to_hemisphere(x, y):
        a = 2 * x - 1
        b = 2 * y - 1

        if (a > -b):
            if (a > b):
                r = a
                phi = (numpy.pi / 4) * (b / a)
            else:
                r = b
                phi = (numpy.pi / 4) * (2 - (a / b))
        else:
            if (a < b):
                r = -a
                phi = (numpy.pi / 4) * (4 + (b / a))
            else:
                r = -b
                if (b != 0):
                    phi = (numpy.pi / 4) * (6 - (a / b))
                else:
                    phi = 0

        u = r * numpy.cos(phi)
        v = r * numpy.sin(phi)

        r2 = u ** 2 + v ** 2
        x = u * numpy.sqrt(2 - r2)
        y = v * numpy.sqrt(2 - r2)
        z = 1 - r2

        return (x, y, z)

    unit_square = [(i, j) for i in numpy.linspace(0, 1, nb_cameras_root) for j in
                   numpy.linspace(0, 1, nb_cameras_root)]
    return [to_hemisphere(p[0], p[1]) for p in unit_square]

def sample_ball(num):
    h = sample_hemisphere(num)
    output = []

    for i in xrange(len(h)):
        output.append(h[i])
        if h[i][2] != 0:
           output.append((h[i][0], h[i][1], -h[i][2]))
    return output


if config.sampling_method == 'b':
    numpy.save(os.path.join(config.metadata_path, "sample"),sample_ball(config.sampling_dim))
elif config.sampling_method == 'h':
    numpy.save(os.path.join(config.metadata_path, "sample"), sample_hemisphere(config.sampling_dim))    


