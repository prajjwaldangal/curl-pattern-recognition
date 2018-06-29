import cv2
import numpy as np

import matplotlib.pyplot as plt

import random
import os

path = "/Users/prajjwaldangal/Documents/cs/summer2018/"
N = 8

# variations:
#   different curl pattern, different hair
#   camera angle
#


def load_preprocess_contours(path, n, inv=True):
    """
    args:
		n : number of image files
		path: path to a the image directory

	returns: list of original images, binary images as well as
			contour lists and canny edges
	"""
    originals = []
    grays = []
    only_hair = []
    conts_ls = []
    canny = []
    for i in range(n):
        img = cv2.imread(path + 'file' + \
                         str(i + 1) + '.png')
        # convert to color intensity to binary image
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if inv:
            ret, thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
        else:
            ret, thresh = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
        # conts_ls is a list of contoured images
        # cv2.CHAIN_APPROX_SIMPLE requires only 4pts to represent a square contour vs
        #	cv2.CHAIN_APPROX_NONE which would require thousands of points
        img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                     cv2.CHAIN_APPROX_SIMPLE)

        # [1 0 -1]                          [1   1  1]
        # [1 0 -1]   for vertical edges,    [0   0  0]  should be for horizontal edges
        # [1 0 -1]                          [-1 -1 -1]
        # gives you nice, point-ful edges
        edges = cv2.Canny(thresh, threshold1=100, threshold2=255, L2gradient=True)

        originals.append(img)
        grays.append(gray_image)
        only_hair.append(thresh)
        conts_ls.append(contours)
        canny.append(edges)
    return originals, grays, only_hair, conts_ls, canny


# plots from a list consisting of four images
# TO-DO: make dim general; plt may not support it
def plotting(ls, fig):
    plt.figure(fig)

    plt.subplot()
    plt.subplot(221)
    plt.imshow(ls[0])

    plt.subplot(222)
    plt.imshow(ls[1])

    plt.subplot(223)
    plt.imshow(ls[2])

    plt.subplot(224)
    plt.imshow(ls[3])
    """
    plt.subplot(425)
    plt.imshow(ls[4])

    plt.subplot(426)
    plt.imshow(ls[5])

    plt.subplot(427)
    plt.imshow(ls[6])

    plt.subplot(428)
    plt.imshow(ls[7])
    """

    plt.show()

#
def blur(ls):
    blurred = []
    for img in ls:
        blur = cv2.blur(img, (10, 10))
        blurred.append(blur)

    return blurred


def plot_imgs(only_hair, path, n, orig=[], gray=[], conts_ls=[], canny=[]):
    i = 0
    conts_imgs = []
    for i in range(n):
        myFile = "{Path}{dir_ini}{ind}{form}".format(Path=path, dir_ini="figures/cont_",
                                                     ind=i + 1, form='.png')
        contrs = cv2.drawContours(only_hair[i], conts_ls[i], -1, (0, 255, 0), 3)
        # cv2.imwrite(myFile, only_hair[i])
        # img = cv2.imread(myFile)

        conts_imgs.append(contrs)
        # if os.path.isfile(myFile):
        #     os.remove(myFile)
        # else:
        #     print("Error: %s file not found" % myFile)
    plotting(only_hair, 1)
    # try plotting(conts_ls, ..)
    plotting(conts_imgs, 2)
    plotting(canny, 3)

    return


# for comparison of different hair types
def plot_hair_types(root):
    # root is the root directory that contains
    # all the other hair types
    hair_types = ["3c hair", "4a hair", "4b hair", "4c hair"]

    # rename image files into uniform numeric with ext
    for i in range(4):
        files = os.listdir(root+"/algo/downloads/"+hair_types[i]+"/")
        l = len(files)
        for i in range(l):
            os.rename(files[i], str(i+1)+ ".png")

    # randomly select filenames between 1 and number of files in dir
    for i in range(4):
        files = os.listdir(root+"/algo/downloads/"+hair_types[i]+"/")
        l = len(files)
        for j in range(2):
            n = random.randint(1, l)
            # os.listdir, os.rename


    for i in range(4):
        pass
        # plot two
    pass


originals, _, only_hair, conts_ls, canny = load_preprocess_contours(path, 4)
# the following line is used to compare original, contours and canny images
plot_imgs(only_hair, path, 4, canny=canny, conts_ls=conts_ls)

# returns image obtained after subtracting image from face_cascade coordinates
def face_cascade():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv2.imread('file1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # intensity image

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # faces is returned as a four number array/tuple, must be
    #       two coordinates that denote the centroid of the bounding
    #       rectangle and two values for height and width of the
    #       bounding rectangle

    return gray - faces

subtracted_img = face_cascade()
cv2.imshow('substracted_img', subtracted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
