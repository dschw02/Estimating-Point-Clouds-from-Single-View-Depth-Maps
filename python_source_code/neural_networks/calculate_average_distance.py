import numpy as np
import os
import graph_utility
import argparse
import net_config_loader
import preloader
import open3d as o3d
import shapenet_class_information as sci

parser = argparse.ArgumentParser(description="")

parser.add_argument('-p', nargs = '?', help = 'path to predicted data')
parser.add_argument('-c', nargs = '?', help = 'class choice')

args = parser.parse_args()

os.chdir(args.p)
os.chdir("..")
print os.getcwd()

net_config = net_config_loader.load_config(os.getcwd() + "/") 
graph_util = graph_utility.graph_utility(net_config, 105, class_choice = args.c)
X_val, Y1_val, Y2_val = graph_util.X_val, graph_util.Y1_val, graph_util.Y2_val
vox_val = Y2_val.p.p.files
vox_val = [v.replace("images", "voxels") for v in vox_val]
vox_val.sort()

pred = [p for p in os.listdir(args.p) if p.endswith(".ply")]
test = lambda x: int(x[:-4])

vox_pred = sorted(pred, key=test)
vox_pred = [args.p + "/" + v for v in vox_pred]

class_sections = Y2_val.p.p.give
split = [0]
for r in class_sections:
	split.append(split[-1] + r[1] - r[0])

classes = ["lamp","plane","wastebin","suitcase","basket","bathtub","bed","bench","birdhouse","bookshelf","bottle","bowl","bus","cabinet","camera","can","cap","car","cellphone","chair","clock","keypad","dishwasher","display","earphone","faucet","file_cabinet","guitar","helmet","jar","knife","laptop","loudspeaker","mailbox","microphone","microwave","motorcycle","mug","piano","pillow","pistol","flowerpot","printer","remote","rifle","projectile","skateboard","sofa","stove","table","telephone","tower","train","watercraft"]

avg_gt = []
avg_p = []
for i in xrange(len(vox_pred)):
	v1 = o3d.io.read_point_cloud(vox_pred[i])
	tmp = (np.asarray(v1.points) + 1.) / 2.
	v1.points = o3d.utility.Vector3dVector(tmp)

	v2 = np.load(vox_val[i])["arr_0"] / 256.
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(np.transpose(v2))

	avg_gt.append(np.mean(v1.compute_point_cloud_distance(pcd)))
	avg_p.append(np.mean(pcd.compute_point_cloud_distance(v1)))
	if i % 100 == 0:
		print i


print "mean over all classes gt -> p: ", np.mean(avg_gt)
print "mean over all classes p -> gt: ", np.mean(avg_p)
print
print "median over all classes gt -> p: ", np.median(avg_gt)
print "median over all classes gt -> p: ", np.median(avg_p)
print 
print "-------------------------------------------------------"
print "-------------------------------------------------------"

for i in xrange(len(classes)):
	print classes[i], "mean gt -> p, mean p -> gt, median gt->p, median p->gt"
	print np.mean(avg_gt[split[i]: split[i+1]]), np.mean(avg_p[split[i]: split[i+1]]), np.median(avg_gt[split[i]: split[i+1]]), np.median(avg_p[split[i]: split[i+1]])
	print "-------------------------------------------------------"
