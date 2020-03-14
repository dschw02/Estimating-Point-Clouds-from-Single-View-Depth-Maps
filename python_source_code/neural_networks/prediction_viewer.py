from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
import numpy
import os

class prediction_viewer:
	def __init__(self, input_data, pred_depth, pred_mask, true_data):
		self.input_imgs = input_data
		self.pred_imgs, self.pred_masks = pred_depth, pred_mask
		self.true_imgs, self.true_masks = numpy.split(true_data, 2, axis = 4)
		self.true_imgs, self.true_masks	= numpy.squeeze(self.true_imgs), numpy.squeeze(self.true_masks)

	def show(self, data):
		img = Image.fromarray(data * 255.)
		img = Image.rotate(90)
		img.show()

	def shape_image(self, data, dim_x, dim_y):
		data = [numpy.rot90(d) for d in data]
		shape = numpy.shape(data)				
		data = numpy.reshape(data, [dim_x, dim_y, shape[-2], shape[-1]])

		data = numpy.hstack([data[i] for i in xrange(dim_x)])
		data = numpy.hstack([data[i] for i in xrange(dim_y)])	

		return data

	def view_all(self, dim_x, dim_y, ind, save = False, save_dir = None, save_name = None):
		img_dim_x, img_dim_y = numpy.shape(self.input_imgs[ind])[0], numpy.shape(self.input_imgs[ind])[1]
		total_width, total_height = dim_y * img_dim_y, dim_x * img_dim_x 

		header_heights = [0.5, 0.4, 0.3, 0.4, 0.3]
		header_text = ['Input Image', 'True Depths', 'Predicted Depths', 'True Masks', 'Predicted_masks']
		headers = [Image.new('F', (total_width, int(total_height * fac)), 225.) for fac in header_heights]

		draws = [ImageDraw.Draw(img) for img in headers]
		for i in xrange(len(headers)):
			draws[i].text((total_width * 0.35, total_height * header_heights[i] * 0.35), header_text[i]) 

		input_img = Image.fromarray(self.input_imgs[ind])
		input_img = input_img.rotate(90)
		input_img = input_img.resize((dim_x * img_dim_x, total_height))
		input_img = ImageOps.pad(input_img, (total_width, total_height))

		true_i = self.shape_image(self.true_imgs[ind], dim_x, dim_y)
		true_m = self.shape_image(self.true_masks[ind], dim_x, dim_y)
		pred_i = self.shape_image(self.pred_imgs[ind], dim_x, dim_y)
		pred_m = self.shape_image(self.pred_masks[ind], dim_x, dim_y)
		data = numpy.vstack([headers[0], numpy.array(input_img) * 255., headers[1], true_i * 255., headers[2], pred_i * 255., headers[3], true_m * 255., headers[4], pred_m * 255.])
		
		if save == False:
			img = Image.fromarray(data)	
			img.show()
		else:
			img = Image.fromarray(data)
			img.convert('L').save(os.path.join(save_dir, str(save_name)+".PNG"))

	def rot90(data):
		return [numpy.rot90(d) for d in data]
		
	