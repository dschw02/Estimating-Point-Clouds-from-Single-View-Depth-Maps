import os
import numpy
import sys
from PIL import Image
import copy
import shapenet_class_information as sci

#custom imports
from find_mount import repository_path
sys.path.append(repository_path)
import config
#--------------

class preloader:
    def __init__(self, directory, selection, class_choice = None):
        self.files = os.listdir(directory)
        self.class_choice = class_choice
        if not any(os.path.isfile(os.path.join(directory, i)) for i in self.files):
        	print "no file in directory"
        	self.files = [os.path.join(directory, f, "data/" + "raw_imgs_npy.npz") for f in self.files]
        else:
        	self.files = [os.path.join(directory, f) for f in self.files]

        self.files.sort()
        print "selection: ", selection

        if not class_choice is None:
            if class_choice in sci.get_classes():
                f, t = "%05d" % (sci.class_dict[class_choice][0]), "%05d" % (sci.class_dict[class_choice][1])
                f = [i for i, path in enumerate(self.files) if f in path][0] 
                t = [i for i, path in enumerate(self.files) if t in path][0]
                self.files = numpy.array(self.files)[xrange(f,t)]
                #subset = sci.get_all(class_choice)
                #self.files = numpy.array(self.files)[subset]
                #self.files = [self.files[i] for i in subset]
            else:
                print "preloader: not a valid class, using all data"

        self.files = [f for f in self.files if os.path.isfile(f)]

        self.selection = selection
        self.length = len(self.files)

    def __getitem__(self, key):
    	try:
            if isinstance(key, int):
                return numpy.load(self.files[key])["arr_0"][self.selection]
            elif(type(key) in [list, numpy.ndarray]):
                return [numpy.load(self.files[item])["arr_0"][self.selection] for item in key]
            else:
                return [numpy.load(item)["arr_0"][self.selection] for item in self.files[key]]
        except:
        	print "error with key:"
        	print key 


    def __len__(self):
        return self.length
        
    def shape(self):
        return numpy.insert(numpy.shape(self[0]), 0, self.length)
 

    def show(self, index):
        for x in self[index]:
            print x
            img = Image.fromarray(x * 255.)
            img = img.rotate(90)
            img.show()

    def split(self, share):
	if not self.class_choice is None:
	    f1, f2 = self.files[:int(self.length * (1 - share))], self.files[int(self.length * (1 - share)):]

	    other = copy.deepcopy(self)

            self.files = f1
            other.files = f2
            self.length = len(self.files)
            other.length = len(other.files)

            return other
	else:
	    classes = sci.class_dict	    
	    sections = []

	    for c in sci.class_dict:
		f, t = "%05d" % (sci.class_dict[c][0]), "%05d" % (sci.class_dict[c][1])
		f = [i for i, path in enumerate(self.files) if f in path][0] 
		t = [i for i, path in enumerate(self.files) if t in path][0]
		sections.append([f,t])
	    sections.sort()

	    keep = [[c[0], c[0] + int((c[1] - c[0]) * (1-share))] for c in sections]
	    give = [[c[0] + int((c[1] - c[0]) * (1-share)), c[1]] for c in sections]	    

	    sel1, sel2 = [], []	    
	    for i in xrange(len(sections)):
		    sel1 = sel1 + self.files[keep[i][0] : keep[i][1]]
		    sel2 = sel2 + self.files[give[i][0] : give[i][1]]

	    other = copy.deepcopy(self)		
        other.give = give
        self.files = sel1
        other.files = sel2
        self.length = len(self.files)
        other.length = len(other.files)

        return other	


class mask_wrapper:
    def __init__(self, preloader, mask_preloader):
        self.p = preloader
        self.mp = mask_preloader
        if len(self.p) != 0:
            self.dim = len(self.p.shape())     
        else:
            self.dim = 0

    def __len__(self):
        return self.p.length

    def __getitem__(self, key):
        images = self.mp[key]
        masks = numpy.array(numpy.greater(images, 0), dtype="float32")
        images = self.p[key]

        if (isinstance(key, int)):
            return numpy.stack([images, masks], axis=self.dim-1)

        return numpy.stack([images, masks], axis=self.dim)

class scaled_preloader:
    def __init__(self, preloader, scaling_value):
        self.p = preloader
        self.scale = scaling_value

    def __len__(self):
        return self.p.length

    def shape(self):
        return self.p.shape()

    #implementation from:
    #https://stackoverflow.com/questions/48121916/numpy-resize-rescale-image
    #based on:
    #https://scipython.com/blog/binning-a-2d-array-in-numpy/
    def __getitem__(self, key):
        ret = numpy.array(self.p[key])
        s = numpy.shape(ret)

        if len(numpy.shape(ret)) == 3:
            ret = ret.reshape((s[0], s[-2] / self.scale, self.scale,
                                     s[-1] / self.scale, self.scale)).max(4).max(2)
        elif len(numpy.shape(ret)) == 2:
            ret = ret.reshape((s[-2] / self.scale, self.scale,
                               s[-1] / self.scale, self.scale)).max(3).max(1)
        else:
            ret = ret.reshape((s[0], s[1], s[-2] / self.scale, self.scale,
                                           s[-1] / self.scale, self.scale)).max(5).max(3)

        return ret
