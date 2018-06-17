import cv2
import numpy as np

import matplotlib.pyplot as plt

import os

path = '/Users/prajjwaldangal/Documents/cs/summer2018/'
N = 8
def load_preprocess_contours(path, n):
	"""
	args:
		n : number of image files
		path: path to a the image directory

	returns: list of original images, binary images as well as
			contour lists
	"""
	originals = []
	grays = []
	only_hair = []
	conts_ls = []
	for i in range(n):
		img = cv2.imread(path+'file' +\
		str(i+1) + '.png')
		originals.append(img)
		# convert to color intensity to binary image
		gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		grays.append(gray_image)
		ret,thresh1 = cv2.threshold(gray_image,100,255,cv2.THRESH_BINARY)
		thresh1_new = thresh1 # findCountours modifies the src image
		only_hair.append(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR))
		# conts_ls is a list of contoured images
		# cv2.CHAIN_APPROX_SIMPLE requires only 4pts vs
		#	cv2.CHAIN_APPROX_NONE
		img2, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, 
			cv2.CHAIN_APPROX_SIMPLE)
		conts_ls.append(contours)
	return originals, grays, only_hair, conts_ls

#originals, only_hair, conts_ls = load_preprocess_and_contours(N, path)



def plotting(ls, fig):
	plt.figure(fig)

	plt.subplot()
	plt.subplot(421)
	plt.imshow(ls[0])

	plt.subplot(422)
	plt.imshow(ls[1])

	plt.subplot(423)
	plt.imshow(ls[2])

	plt.subplot(424)
	plt.imshow(ls[3])

	plt.subplot(425)
	plt.imshow(ls[4])

	plt.subplot(426)
	plt.imshow(ls[5])

	plt.subplot(427)
	plt.imshow(ls[6])

	plt.subplot(428)
	plt.imshow(ls[7])

	plt.show()

#plotting(only_hair, 1)

def contour_plot_imgs(only_hair, conts_ls, path, n):
	i=0
	imgs = []
	for i in range(n):
		myFile = path+'figures/cont_'+str(i)+'.png'
		cv2.drawContours(only_hair[i], conts_ls[i], -1, (0,255,0), 3)
		cv2.imwrite(myFile, only_hair[i])
		img = cv2.imread(myFile)
		imgs.append(img)
		
		if os.path.isfile(myFile):
			os.remove(myFile)
		else:
			print("Error: %s file not found" % myFile)
		#cv2.imshow("Keypoints", only_hair[i])
		#cv2.waitKey(0)
	return imgs

#imgs = contour_plot_imgs(path)

#plotting(imgs, 2)
# contours part --> will also have plots for images
