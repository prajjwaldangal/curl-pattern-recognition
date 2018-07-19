import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import plot_lib as plt2

import random
import os

import argparse

# TO-DO: mention path explicitly inside each function

# variations:
#   different curl pattern, different hair
#   camera angle
#

hair_types = ["3c", "4a", "4b", "4c"]

# manage cline argument
# batch load in tf
parser = argparse.ArgumentParser()
parser.add_argument('-i','--input-dir', default='/Users/prajjwaldangal/Documents/cs/summer2018/algo/downloads/' ,
                    help='Directory containing images (png format) to process')
args = parser.parse_args()
# load images into the program
def load_preprocess_contours(hair_type, n, inv=True):
    """
    args:
		hair_type : the hair type
		n : number of image files
		inv : whether or not to invert the binary image, for our algorithm we want it to be True

	returns: list of binary images, original images, grays as well as
			contour lists and canny edges
	"""
    root = args.input_dir
    j = hair_types.index(hair_type.strip().lower())
    originals = []
    grays = []
    only_hair = []
    conts_ls = []
    canny = []

    for i in range(n):
        path = os.path.join(root, hair_types[j], str(i + 1)+'.png')
        print(path)
        img = cv.imread(path)
        # convert to color intensity image to binary image
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if inv:
            ret, thresh = cv.threshold(gray_image, 100, 255, cv.THRESH_BINARY_INV)
        else:
            ret, thresh = cv.threshold(gray_image, 100, 255, cv.THRESH_BINARY)
        # conts_ls is a list of contoured images
        # cv.CHAIN_APPROX_SIMPLE requires only 4pts to represent a square contour vs
        #	cv.CHAIN_APPROX_NONE which would require thousands of points
        img2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                                     cv.CHAIN_APPROX_SIMPLE)

        # [1 0 -1]                          [1   1  1]
        # [1 0 -1]   for vertical edges,    [0   0  0]  should be for horizontal edges
        # [1 0 -1]                          [-1 -1 -1]
        # gives you nice, pointful edges
        edges = cv.Canny(thresh, threshold1=100, threshold2=255, L2gradient=True)

        originals.append(img)
        grays.append(gray_image)
        only_hair.append(thresh)
        conts_ls.append(contours)
        canny.append(edges)
    return only_hair, grays, originals, conts_ls, canny

# returns image obtained after subtracting image from face_cascade coordinates
def face_cascade():
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv.imread('file1.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # intensity image

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # faces is returned as a four number array/tuple, must be
    #       two coordinates that denote the centroid of the bounding
    #       rectangle and two values for height and width of the
    #       bounding rectangle.
    # It looks like: [[256 209 264 264]]
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #print(faces)
    x,y,w,h = faces[0]
    # set the face coordinates to o
    for i in range(y-h, y+h+1):
        for j in range(x-w, x+w+1):
            gray[i][j] = 0
    #gray[x-w:x+w+1][y-h:y+h+1] = 0
    #ret, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)

    #return gray - faces
    return gray

# subtracted_img = face_cascade()
# cv.imshow('substracted_img', subtracted_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# plots from a list consisting of four images
# TO-DO: make dim general; plt may not support it
def plotting(ls, fig):
    """
    Plots four images i.e. ls should have four members
    :param ls: list of images
    :param fig: figure number
    :return: NA
    """
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
    """

    :param ls: list of images
    :return: blurred list
    """
    blurred = []
    for img in ls:
        blur = cv.blur(img, (10, 10))
        blurred.append(blur)

    return blurred

#
# Note: 3c is a cleaned directory
#

# Specifically, compare original, contours and canny images.
# TO-DO: add hair_type argument and changes
# Spaghetti code
def plotting2(only_hair, path, n, orig=[], gray=[], conts_ls=[], canny=[]):
    i = 0
    conts_imgs = []
    for i in range(n):
        # myFile = "{Path}{dir_ini}{ind}{form}".format(Path=path, dir_ini="figures/cont_",
        #                                              ind=i + 1, form='.png')
        contrs = cv.drawContours(only_hair[i], conts_ls[i], -1, (0, 255, 0), 3)
        conts_imgs.append(contrs)
    plotting(only_hair, 1)
    # try plotting(conts_ls, ..)
    plotting(conts_imgs, 2)
    plotting(canny, 3)

    return

# testing plot_lib function (batch plotting)
only_hair, grays, _, _, _ = load_preprocess_contours("3c", 100, inv=True)
# plt2.plotting(grays, "3c")
# plt2.plotting(only_hair, "3c")


# run this function once to rename image files
# into numeric filenames like, 3c/1.png, 3c/2.png, 4a/1.png, 4a/2.png ...
# already ran on my machine, so I don't have to worry about it
def batch_rename(hair_dir, extra=""):
    # rename image files into uniform numeric with exts
    path = os.path.join(args.input_dir, hair_dir)
    filenames = os.listdir(path)
    for idx, filename in enumerate(filenames):
        os.rename(os.path.join(path, filename), os.path.join(path, extra+str(idx+1)+".png"))

# batch_rename("3c")

"""
######### Model histogram and meanshift  #############
s = [[1,2,3],[1,2,3],[1,2,3]]
calcHist(s) returns  [3,3,3]
calcBackProject(s, hist, 2) returns  [[6,6,6],[6,6,6],[6,6,6]]
hist = cv2.calcHist([s], [0], None, [3], [1,4])
dst = cv2.calcBackProject([s], [0], hist, [1,4], 2)

meanshift -->  calcBackProject  -->  calcHistogram
scale
"""

def resize(out_size, hair_type):
    pass

def mean_shift(hair_type, n):
    """

    :param path: path to image files
    :param n: number of images to segment
    :return: list of segmented images
    """
    root = args.input_dir
    imgs = []
    r,h,c,w = 250,90,400,125  # simply hardcoded the values
    track_window = (c,r,w,h)
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
    for i in range(n):
        img = cv.imread(os.path.join(root, hair_type, str(n)+".png"))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # applyColorMap applies a GNU Octave/MATLAB equivalent colormap on a given image
        # cv.applyColorMap(src, colormap [, dst]) -> dst
        ret, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
        # apply cv.calcHist and cv.calcBackProject on thresh first of all
        hist = cv.calcHist([gray], [0], None, [255-100], [100, 255])
        dst = cv.calcBackProject([gray], [0], hist, [100, 255], 2)
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        x,y,w,h = track_window
        img2 = cv.rectangle(img, (x,y), (x+w,y+h), 255,2)
        imgs.append(img2)
        # cv.imshow('img2',img2)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        return imgs

# imgs = mean_shift("/Users/prajjwaldangal/Documents/cs/summer2018/algo/downloads/4c", 50)
# print("Shape: {} x {}".format(len(imgs), len(imgs[0])))
# plt2.plotting(imgs)
"""

1. get only hair part
2. try to get most of the hair, as much as is possible without letting non-hair part into the segmented image
3. it's okay if some skin portion, background or any thing else present in the segmented image. Do try your best to 
    minimize it though.
4. don't worry too much about loose hair. Loose hair is the thinly-dense hair on the contours of the head. Get however
    much you can of loose hair without compromising the above points.
5. Variation of shades is present in the images between different classes. Also curls on the end of hair bundle could
    help differentiate between classes.

# flixify start from 30 and onwards
###################### Mean shift segmentation method #####################
Histograms are collected counts of data organized into a set of predefined bins
intensity values (or ranges of intensity values)  on x axis, frequency on y axis for a given (a x b) region of img

Monday meeting issues to raise: add as collaborators on github, 
                                somebody needs to look into k-means clustering for segmentation
                                
"""
