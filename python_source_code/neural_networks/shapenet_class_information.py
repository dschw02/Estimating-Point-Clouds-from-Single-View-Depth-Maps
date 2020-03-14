import sys
import random

class_dict = dict(lamp = [ 0 ,  2318 ],
		plane = [ 2318 ,  6363 ],
		wastebin = [ 6363 ,  6706 ],
		suitcase = [ 6706 ,  6789 ],
		basket = [ 6789 ,  6902 ],
		bathtub = [ 6902 ,  7758 ],
		bed = [ 7758 ,  7991 ],
		bench = [ 7991 ,  9804 ],
		birdhouse = [ 9804 ,  9877 ],
		bookshelf = [ 9877 ,  10329 ],
		bottle = [ 10329 ,  10827 ],
		bowl = [ 10827 ,  11013 ],
		bus = [ 11013 ,  11952 ],
		cabinet = [ 11952 ,  13523 ],
		camera = [ 13523 ,  13636 ],
		can = [ 13636 ,  13744 ],
		cap = [ 13744 ,  13800 ],
		car = [ 13800 ,  17333 ],
		cellphone = [ 17333 ,  18164 ],
		chair = [ 18164 ,  24942 ],
		clock = [ 24942 ,  25593 ],
		keypad = [ 25593 ,  25658 ],
		dishwasher = [ 25658 ,  25751 ],
		display = [ 25751 ,  26844 ],
		earphone = [ 26844 ,  26917 ],
		faucet = [ 26917 ,  27661 ],
		file_cabinet = [ 27661 ,  27959 ],
		guitar = [ 27959 ,  28756 ],
		helmet = [ 28756 ,  28918 ],
		jar = [ 28918 ,  29514 ],
		knife = [ 29514 ,  29938 ],
		laptop = [ 29938 ,  30398 ],
		loudspeaker = [ 30398 ,  31995 ],
		mailbox = [ 31995 ,  32089 ],
		microphone = [ 32089 ,  32156 ],
		microwave = [ 32156 ,  32308 ],
		motorcycle = [ 32308 ,  32645 ],
		mug = [ 32645 ,  32859 ],
		piano = [ 32859 ,  33098 ],
		pillow = [ 33098 ,  33194 ],
		pistol = [ 33194 ,  33501 ],
		flowerpot = [ 33501 ,  34103 ],
		printer = [ 34103 ,  34269 ],
		remote = [ 34269 ,  34335 ],
		rifle = [ 34335 ,  36708 ],
		projectile = [ 36708 ,  36793 ],
		skateboard = [ 36793 ,  36945 ],
		sofa = [ 36945 ,  40118 ],
		stove = [ 40118 ,  40336 ],
		table = [ 40336 ,  48772 ],
		telephone = [ 48772 ,  49861 ],
		tower = [ 49861 ,  49994 ],
		train = [ 49994 ,  50383 ],
		watercraft = [ 50383 ,  52322 ]
)

def get_all(class_name):
    class_range = class_dict[class_name]
    return xrange(class_range[0], class_range[1])

def get_random(class_name):
    class_range = class_dict[class_name]
    return random.randrange(class_range[0], class_range[1])    

def get_classes():
    return class_dict.keys()

def get_class(model_number):
    for c in get_classes():
		class_range = class_dict[c]    
		class_range = xrange(class_range[0], class_range[1])	

		if model_number in class_range:
			return c 



