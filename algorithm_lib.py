import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

import random
import os

N = 8

# variations:
#   different curl pattern, different hair
#   camera angle
#

hair_types = ["3c", "4a", "4b", "4c"]

# load images into the program
def load_preprocess_contours(root, hair_type, n, inv=True):
    """
    args:
        root: absolute path to the root directory that contains image data for all hair types
		hair_type : the hair type
		n : number of image files
		inv : whether or not to invert the binary image, for our algorithm we want it to be True

	returns: list of original images, binary images as well as
			contour lists and canny edges
	"""
    j = hair_types.index(hair_type.strip().lower())
    originals = []
    grays = []
    only_hair = []
    conts_ls = []
    canny = []
    for i in range(n):
        img = cv.imread(os.path.join(root, hair_types[j] + " hair", str(i + 1)+'.png'))
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
def plotting2(only_hair, path, n, orig=[], gray=[], conts_ls=[], canny=[]):
    i = 0
    conts_imgs = []
    for i in range(n):
        myFile = "{Path}{dir_ini}{ind}{form}".format(Path=path, dir_ini="figures/cont_",
                                                     ind=i + 1, form='.png')
        contrs = cv.drawContours(only_hair[i], conts_ls[i], -1, (0, 255, 0), 3)
        # cv.imwrite(myFile, only_hair[i])
        # img = cv.imread(myFile)

        conts_imgs.append(contrs)
    plotting(only_hair, 1)
    # try plotting(conts_ls, ..)
    plotting(conts_imgs, 2)
    plotting(canny, 3)

    return


# run this function once to rename image files
# into numeric filenames like, 3c/1.png, 3c/2.png, 4a/1.png, 4a/2.png ...
# already ran on my machine, so I don't have to worry about it
def batch_rename(hair_type, root="/Users/prajjwaldangal/Documents/cs/summer2018/algo/downloads/"):
    # root : root directory that contains all the other hair types
    #       "/Users/prajjwaldangal/Documents/cs/summer2018/algo/downloads/"

    # rename image files into uniform numeric with ext
    pass

## plot 400 images, 100 images for each class
def plotting3(class):
    root = "/Users/prajjwaldangal/Documents/cs/summer2018/algo/downloads"
    _, grays, _, _, _ = load_preprocess_contours(root, class, 100, inv=True)
    


# 1. Run batch_rename once
# 2. Run load preprocess contours with the appropriate path.
path = "/Users/prajjwaldangal/Documents/cs/summer2018/algo/downloads"
only_hair_ls, grays, originals, conts_ls, canny_ls = load_preprocess_contours(
    "/Users/prajjwaldangal/Documents/cs/summer2018/algo/downloads", "4b", 4)
plotting2(only_hair_ls, path, 4, orig=originals, gray=grays, conts_ls=conts_ls, canny=canny_ls)

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

subtracted_img = face_cascade()
cv.imshow('substracted_img', subtracted_img)
cv.waitKey(0)
cv.destroyAllWindows()

"""
######### Model histogram and meanshift  #############
s = [[1,2,3],[1,2,3],[1,2,3]]
calcHist(s) returns  [3,3,3]
calcBackProject(s, hist, 2) returns  [[6,6,6],[6,6,6],[6,6,6]]
hist = cv2.calcHist([s], [0], None, [3], [1,4])
dst = cv2.calcBackProject([s], [0], hist, [1,4], 2)
"""

from sklearn.cluster import MeanShift

# meanshift -->  calcBackProject  -->  calcHistogram
# scale

def mean_shift(path, n):
    """

    :param path: path to image files
    :param n: number of files to segment
    :return: segmented files
    """
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )
    for i in range(n):
        img = cv2.imread(os.path.join(path, str(n)+".png"))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # applyColorMap applies a GNU Octave/MATLAB equivalent colormap on a given image
        # cv2.applyColorMap(src, colormap [, dst]) -> dst
        ret, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
        # apply cv2.calcHist and cv2.calcBackProject
        hist = cv2.calcHist([gray], [0], None, [255-100], [100, 255])
        ret, track_window = cv.meanShift(dst, track_window, term_crit)

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
