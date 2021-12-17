import os
import numpy as np
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import imageio

def shake_imgs(path, path_shaked_imgs, shift):
	ret = os.access(path, os.F_OK)

	if ret:
		f_list = sorted(os.listdir(path))

	num_images = 2

	for j in range(0,len(f_list)):
		if (os.path.isdir(path_shaked_imgs + '/' + f_list[j].split('.')[0]) == False):
			os.mkdir(path_shaked_imgs + '/' + f_list[j].split('.')[0])
		for img in range(0,  num_images):

			with cbook.get_sample_data(path + '/' + f_list[j]) as image_file:
				image = plt.imread(image_file)

				# from top to bottom
				if img==0:
					img_new = np.zeros(image.shape)
					img_new[0:(shift+1), :] = image[0:(shift+1), :]
					img_new[(shift+1):-1:1, :] = image[0:-(shift + 2):1, :]
				# from left to rigth
				elif img==1:
					img_new = np.zeros(image.shape)
					img_new[:, 0:(shift+1)] = image[:, 0:(shift+1)]
					img_new[:, (shift + 1):-1:1] = image[:, 0:-(shift + 2):1]

			imageio.imwrite(path_shaked_imgs + '/' + f_list[j].split('.')[0] + '/' + str(img) + '.jpg', img_new)


def cross_corr():
	pass

if __name__ == "__main__":

	path = '/Tesi/images/'
	path_shaked_imgs = '/Tesi/shaked_imgs'

	shift = 1

	shake_imgs(path, path_shaked_imgs,shift)

print('end')
