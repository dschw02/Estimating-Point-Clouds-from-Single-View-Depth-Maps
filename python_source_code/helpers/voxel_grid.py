import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import os
from PIL import Image
import sys
import gc
import open3d as o3d
#import trimesh

class voxel_grid:
    eps = 0.001
    dim = 0
    upscale_factor = 0
    viewed_points = 0
    x, y, z = [], [], []
    grid = numpy.array([])
    quaternions = None

    '''
	INPUT:



	'''
    def __init__(self, input, viewed_points=10000, dim=256, eps=0.001, upscale_factor=0, scale = True):
        self.eps = eps

        if type(input) == str:
            if not os.path.isfile(input):
                print "not a file"
            elif input.endswith(".npy"):
                self.load_npy(input)
            elif input.endswith(".npz"):
                self.load_npz(input)
            elif input.endswith(".obj"):
                self.rasterize_from_path(input, dim)
            elif input.endswith(".binvox"):
                self.load_binvox(input)
            else:
                print "unknown filetype"
        elif isinstance(input, numpy.ndarray) or isinstance(input, list):
            if len(numpy.shape(input)) == 3:
                self.grid = input
                self.x, self.y, self.z = numpy.nonzero(input)
            elif len(input) == 3:
                self.x, self.y, self.z = input[0], input[1], input[2]
            else:
                print "neither 3d grid nor sparse grid"
        else:
            "no known functionality"

        self.upscale_factor = upscale_factor
        self.dim = dim
        for i in xrange(upscale_factor):
            self.upscale(upscale_factor)
		
        if scale:
	   	    self.scale()
        self.viewed_points = viewed_points


    '''
	WARNING: |imgs| == |mats|
	Projects "imgs" into 3d space using inverse projection matrices "mats"
	Only considers depth values > "mask_threshold"
    '''
    def project_to_3d(self, imgs, mats, mask_threshold = 0.1):
        points = []
        for i in xrange(len(mats)):
        	#converting image to quaternions
            val = numpy.reshape(imgs[i], [-1])
            x_ind, y_ind = numpy.meshgrid(numpy.arange(0,numpy.shape(imgs[0])[-2]), numpy.arange(0,numpy.shape(imgs[0])[-1]), indexing='ij')
            x_ind, y_ind = numpy.reshape(x_ind, [-1]), numpy.reshape(y_ind, [-1])
            ones = numpy.ones_like(val)

            #creating mask according to "mask_threshold"
            mask = val > mask_threshold

            #filter valid quaternions and stacking of indices and values 
            val, x_ind, y_ind, ones = val[mask], x_ind[mask], y_ind[mask], ones[mask] 
            img = numpy.stack([x_ind, y_ind, val, ones], axis = 0)

            #adding projected image to the point set representing the 3D model
            points.append(numpy.matmul(mats[i], img))

        #merging the points into a single numpy array
        points = numpy.concatenate(points, axis = 1)
        points = numpy.transpose(numpy.array(points))

        #writing to the objects data
        self.x, self.y, self.z, _ = numpy.split(numpy.transpose(points), 4)
        self.x, self.y, self.z = numpy.reshape(self.x, [-1]), numpy.reshape(self.y, [-1]), numpy.reshape(self.z, [-1])

    '''
	Loads 3D point cloud from binvox file
	Stores grid and nonzeros in object
    '''
    def load_binvox(self, path):
    	#src: https://github.com/czq142857/BAE-NET/blob/master/point_sampling/binvox_rw.py
        def read_header(fp):
            line = fp.readline().strip()
            if not line.startswith(b'#binvox'):
                raise IOError('Not a binvox file')
            dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
            translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
            scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
            line = fp.readline()
            return dims, translate, scale

        #src: https://github.com/czq142857/BAE-NET/blob/master/point_sampling/binvox_rw.py
        def read_as_3d_array(fp, fix_coords=True):
            dims, translate, scale = read_header(fp)
            raw_data = numpy.frombuffer(fp.read(), dtype=numpy.uint8)

            values, counts = raw_data[::2], raw_data[1::2]
            data = numpy.repeat(values, counts).astype(numpy.bool)
            data = data.reshape(dims)

            data = numpy.transpose(data, (0, 1, 2))

            return data

        with open(path, 'rb') as f:
            grid = read_as_3d_array(f)

        self.grid = grid
        self.x, self.y, self.z = numpy.nonzero(grid)

    '''
	Uses open3d to display point cloud contents
	Save and show features independently selectable
    '''
    def to_mesh(self, show = False, save = None):
        xyz = numpy.zeros((numpy.size(self.x), 3))
        
        xyz[:, 0] = numpy.reshape(self.x, -1)
        xyz[:, 1] = numpy.reshape(self.y, -1)
        xyz[:, 2] = numpy.reshape(self.z, -1)

        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        #pcd = pcd.voxel_down_sample(voxel_size=0.08)

        if show:
            o3d.visualization.draw_geometries([pcd])

        if not save is None:
            o3d.io.write_point_cloud(save, pcd)
    

    '''
    INPUT:
    path - Path to wavefront object
    dim - size of the voxel cube that will be saved
    '''        
    def rasterize_from_path(self, path, dim):
        def load_mesh(path):
            f = open(path, "r")

            data = f.read().splitlines()

            # reads all vertices and casts them to 3d-numpy-arrays
            vertices = [numpy.array([float(line.split(" ")[1]), float(line.split(" ")[2]), float(line.split(" ")[3])])
                        for line in data if line.startswith("v ")]

            diff = - (numpy.amin(vertices, axis=0) + numpy.amax(vertices, axis=0)) / 2
            vertices = vertices + diff

            # reads all faces and casts them to 3d-numpy-arrays
            triangles = [numpy.array([int(line.split(" ")[2].split("/")[0]), int(line.split(" ")[3].split("/")[0]),
                                      int(line.split(" ")[4].split("/")[0])])
                         for line in data if line.startswith("f ")]

            # representation of the object as a set of triangles in their given order
            mesh = numpy.array(
                [numpy.array([vertices[t[0] - 1], vertices[t[1] - 1], vertices[t[2] - 1]]) for t in triangles])

            # mapping of the mesh to a discrete grid of voxels
            mesh = (mesh * dim).astype('int')

            # computes bounding boxes of all triangles as [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            bounding_boxes = numpy.array([numpy.array(
                [[numpy.min([triangle[0][0], triangle[1][0], triangle[2][0]]),
                  numpy.min([triangle[0][1], triangle[1][1], triangle[2][1]]),
                  numpy.min([triangle[0][2], triangle[1][2], triangle[2][2]])],

                 [numpy.max([triangle[0][0], triangle[1][0], triangle[2][0]]),
                  numpy.max([triangle[0][1], triangle[1][1], triangle[2][1]]),
                  numpy.max([triangle[0][2], triangle[1][2], triangle[2][2]])]
                 ])
                for triangle in mesh])

            return mesh, bounding_boxes

		#Gets one bounding box per triangle in mesh and returns voxel model
        def rasterize_mesh(mesh, bounding_boxes, dim):
            def rasterize_triangle(bounding_box, triangle, dest, dim):
            	#checks if a triangle is not on a line
                def is_triangle(triangle):
                    return numpy.linalg.det(triangle) != 0

                #checks all points in a line ([starting point, directional vector]) 
                #and marks voxels it passes 
                def fill_line(line, dest, dim):
                    ld = line[1] - line[0]
                    ld = ld / numpy.linalg.norm(ld)

                    lp = line[0]

                    offset = [dim / 2, dim / 2, dim / 2]

                    while numpy.linalg.norm(lp - line[1]) > 1.:
                        ind1 = numpy.floor(lp).astype('int') + offset
                        ind2 = numpy.ceil(lp).astype('int') + offset

                        if numpy.min(ind1) < 0:
                            print ind1 - offset
                        if numpy.max(ind1) < dim:
                            dest[ind1[0]][ind1[1]][ind1[2]] = 1
                        if numpy.max(ind2) < dim:
                            dest[ind2[0]][ind2[1]][ind2[2]] = 1
                        lp = lp + ld

                    ind1 = numpy.floor(line[1]).astype('int') + offset
                    ind2 = numpy.ceil(line[1]).astype('int') + offset
                    if numpy.max(ind1) < dim:
                        dest[ind1[0]][ind1[1]][ind1[2]] = 1
                    if numpy.max(ind2) < dim:
                        dest[ind2[0]][ind2[1]][ind2[2]] = 1

                #calculates intersections between "triangle" and "plain"
                def intersect_triangle_plane(triangle, plane):
                	#calculates intersections between "line" and "plane"
                    def intersect_line_plane(line, plane):
                        line_scale = numpy.linalg.norm(line[1])

                        #directional vector has length 0
                        if numpy.linalg.norm(line[1]) < self.eps:
                            return None
                        #normalizing directional vector of "line" and normal vector of "plane"
                        else:
                            lp = line[0]
                            ld = line[1] / numpy.linalg.norm(line[1])

                            pp = plane[0]
                            pn = plane[1] / numpy.linalg.norm(plane[1])

                        # "line" is parallel to "plane"
                        if numpy.abs(numpy.dot(ld, pn)) < self.eps:
                            return None

                        # t represents timestep where "line" hits "plane"
                        t = (numpy.dot(pp, pn) - numpy.dot(lp, pn)) / numpy.dot(ld, pn)

                        #collision on triangle?
                        if numpy.linalg.norm(t * ld) / line_scale < 1 + self.eps and t >= 0:
                            return lp + ld * t

                    #points that span triangle
                    a, b, c = triangle[0], triangle[1], triangle[2]

                    #lines between points of the triangle
                    lines = [[a, b - a], [b, c - b], [c, a - c]]

                    #all line-plane intersections + filtering None values
                    intersections = [intersect_line_plane(line, plane) for line in lines]
                    intersections = [inter for inter in intersections if not inter is None]

                    if intersections != []:
                        return numpy.unique([inter for inter in intersections if not inter is None], axis=0)
                    else:
                        return intersections

                # a, b, c - points spanning triangle, tn - triangle normal 
                a, b, c = triangle[0], triangle[1], triangle[2]
                tn = numpy.cross(b - a, c - a)

                #plane normal
                pn = [0, 0, 1]

                #when triangle lies in the chosen plane, alter direction of progress
                if not (numpy.linalg.norm(tn) < self.eps):
                    if numpy.abs(numpy.abs(numpy.arccos(
                            numpy.dot(pn, tn) / (numpy.linalg.norm(pn) * numpy.linalg.norm(tn)))) - numpy.pi) < self.eps:
                        pn = [0, 1, 0]
                else:
                    return


                ind = numpy.argmax(pn)
                pp = bounding_box[0]

                #calculating all intersection of planes in bounding box and triangle
                for layer in xrange(bounding_box[0][ind], bounding_box[1][ind] + 1):
                    if is_triangle(triangle):
                        intersection = intersect_triangle_plane(triangle, [pp, pn])
                    else:
                        if numpy.linalg.norm(a - b) < self.eps:
                            intersection = (a - c) / numpy.linalg.norm(a - c)
                        else:
                            intersection = (a - b) / numpy.linalg.norm(a - b)

                    if len(intersection) == 2:
                        fill_line(intersection, dest, dim)

                    #progressing plane
                    pp = pp + pn

            #output voxel grid initialization
            voxel_model = numpy.zeros((dim, dim, dim))

            #rasterize all triangles in "mesh"
            for i in xrange(len(mesh)):
                rasterize_triangle(bounding_boxes[i], mesh[i], voxel_model, dim)

            return voxel_model

        mesh, bounding_boxes = load_mesh(path)

        voxels = rasterize_mesh(mesh, bounding_boxes, dim)

        self.grid = voxels
        self.x, self.y, self.z = numpy.nonzero(self.grid)

    '''
    initialising objects grid using its nonzeros
    '''
    def init_grid(self, shape):
        self.grid = numpy.zeros(shape)
        for i in len(self.x):
            self.grid[self.x[i]][self.y[i]][self.z[i]] = 1

    '''
    3d max poooling of the grid in the object
    '''
    def downscale_grid(self, blocksize):
        if self.grid.size == 0:
            print "grid not initialized"
        else:
            x_part, y_part, z_part = numpy.array(self.grid.shape) / blocksize
            self.grid = self.grid.reshape(x_part, blocksize, y_part, blocksize, z_part, blocksize).max((1, 3, 5))
            self.x, self.y, self.z = numpy.nonzero(self.grid)

    '''
    Converting point cloud to voxel grid of given dimension
    threshold value determines how many points have to be in the bucket to count
    '''
    def voxelize(self, threshold = 1, dim = 64):
        grid = numpy.zeros([dim, dim, dim])
        #scaling nonzeros to fit [0,dim]
        mi, ma = numpy.min([self.x, self.y, self.z]), numpy.max([self.x, self.y, self.z])
        view_x, view_y, view_z = (self.x - mi) / (abs(mi) + abs(ma)) * dim, (self.y - mi) / (abs(mi) + abs(ma)) * dim, (self.z - mi) / (abs(mi) + abs(ma)) * dim

        for i in xrange(len(view_x)):
        	#incrementing the corresponding bucket to point i
            if view_x[i] < dim and view_x[i] >= 0 and view_y[i] < dim and view_y[i] >= 0 and view_z[i] < dim and view_z[i] >= 0:
                grid[int(view_x[i]), int(view_y[i]), int(view_z[i])] += 1
        
        #writing back to nonzeros using "threshold"    
        view_x, view_y, view_z = [], [], []
        for i in xrange(dim):
            for j in xrange(dim):
                for k in xrange(dim):
                    if grid[i,j,k] > threshold:
                        view_x.append(i)
                        view_y.append(j)
                        view_z.append(k)

        self.grid = grid
        self.x, self.y, self.z = view_x, view_y, view_z

    '''
	Plotting the point cloud using mtplotlib
    '''
    def show(self, dims = [-1,1]):
        view_x, view_y, view_z = self.x, self.y, self.z

        #limiting number of viewed points for performance
        while len(view_x) > self.viewed_points:
            view_x, view_y, view_z = view_x[0::2], view_y[0::2], view_z[0::2]

        fig = pyplot.figure()
        ax = Axes3D(fig)

        ax.set_xlim3d(dims[0], dims[1])
        ax.set_ylim3d(dims[0], dims[1])
        ax.set_zlim3d(dims[0], dims[1])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.scatter(view_x, view_y, view_z, zdir='z')
        pyplot.show()

    '''
   	saves nonzeros to path as a numpy array. 
   	currently sparse is the only mode
    '''
    def save(self, path, mode="sparse"):
        if len(self.x) == 0 and len(self.y) == 0 and len(self.z) == 0:
            print "grid is empty"
        elif len(self.x) != len(self.y) or len(self.y) != len(self.z):
            print "grid information has errors"
        else:
            if mode == "sparse":
                numpy.save(path, numpy.stack([self.x, self.y, self.z]), )

    '''
    loads numpy array (.npy) from path
    automatically detects format (nonzeros/grid)
    '''            
    def load_npy(self, path):
        if not os.path.isfile(path) or not path.endswith(".npy"):
            print path, " does not lead to a .npy file"
        else:
            grid = numpy.load(path)
            if len(grid) == 3:
                self.x, self.y, self.z = grid[0], grid[1], grid[2]
            if len(numpy.shape(grid)) == 3:
                self.grid = grid
                self.x, self.y, self.z = numpy.nonzero(self.grid)

    '''
	loads numpy array (.npy) from path
    automatically detects format (nonzeros/grid)
    '''
    def load_npz(self, path):
        if not os.path.isfile(path) or not path.endswith(".npz"):
            print path, " does not lead to a .npz file"
        else:
            grid = numpy.load(path)
            grid = grid['arr_0']
            if len(grid) == 3:
                self.x, self.y, self.z = grid[0], grid[1], grid[2]
            if len(numpy.shape(grid)) == 3:
                self.grid = grid
                self.x, self.y, self.z = numpy.nonzero(self.grid)

    '''
    getter for the data in grid-form
    '''
    def get_grid(self):
        if self.grid.size == 0:
            self.init_grid((max(self.x) - min(self.x), max(self.y) - min(self.y), max(self.z) - min(self.z)))
        return self.grid

    '''
	scales the stored nonzeros inside the unit cube     
    '''
    def scale(self):
        self.x, self.y, self.z = self.x - (min(self.x)), self.y - (min(self.y)), self.z - (min(self.z))
        self.x, self.y, self.z = self.x - (max(self.x) / 2), self.y - (max(self.y) / 2), self.z - (max(self.z) / 2)
        self.x, self.y, self.z = self.x / (self.dim - 1.0), self.y / (self.dim - 1.0), self.z / (self.dim - 1.0)

        scale = numpy.max(numpy.amax([self.x, self.y, self.z], axis=1))
        max_value = 1. / numpy.sqrt(2)
        scale = scale / max_value
        self.x, self.y, self.z = self.x / scale, self.y / scale, self.z / scale

    '''
    doubles dimension of the grid and creates new points to simulate growth.
    repeats n times
    '''
    def upscale(self, n):
        grid = numpy.stack([self.x, self.y, self.z]).transpose()

        for i in xrange(n):
            grid = grid * 2

            new_points = [[xyz, xyz + [1, 1, 1], xyz + [1, 1, 0], xyz + [1, 0, 1], xyz + [0, 1, 1], xyz + [1, 0, 0],
                           xyz + [0, 1, 0], xyz + [1, 0, 0]] for xyz in grid]
            shape = numpy.shape(new_points)
            grid = numpy.reshape(new_points, (shape[0] * shape[1], 3))

            self.x, self.y, self.z = numpy.transpose(grid)
            self.dim = self.dim * 2 ** (self.upscale_factor - 1)

    '''
    renders the point cloud pointswise into an image of size (img_dim_x, img_dim_y)
    '''
    def render(self, P, img_dim_x, img_dim_y):
    	#conversion to quaternions if necessary
        if self.quaternions is None:
            self.quaternions = numpy.stack([self.x, self.y, self.z, numpy.ones_like(self.x)])

        points = self.quaternions
        out = numpy.zeros((img_dim_x, img_dim_y))
        #rendering
        points = numpy.transpose(numpy.matmul(P, points))

        x, y, z = numpy.split(points, 3, axis=1)
        x, y = x.astype('int'), y.astype('int')
	
	
	mask = (x >= 0) & (x < img_dim_x) & (y >= 0) & (y < img_dim_y)	
	x, y, z = x[mask], y[mask], z[mask]

	for i in xrange(len(x)):
	    out[x[i], y[i]] = max(out[x[i], y[i]], z[i])	

        #z buffer
        #for i in xrange(len(points)):
        #    if x[i] >= 0 and x[i] < img_dim_x and y[i] >= 0 and y[i] < img_dim_y:
        #        if out[x[i], y[i]] < z[i]:
        #            out[x[i], y[i]] = z[i]

        return out
