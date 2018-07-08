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

hair_types = ["3c", "4a", "4b", "4c"]

# load images into the program
def load_preprocess_contours(root, n, hair_type="3c", inv=True):
    """
    args:
        root: absolute path to the root directory that contains image data for all hair types
		n : number of image files
		hair_type : the hair type
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
        img = cv2.imread(os.path.join(root, hair_types[j], str(i + 1), '.png'))
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
        # gives you nice, pointful edges
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
        blur = cv2.blur(img, (10, 10))
        blurred.append(blur)

    return blurred

# Specifically, compare original, contours and canny images.
# TO-DO: add hair_type argument and changes
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


# run this function once to rename image files
# into numeric filenames like, 3c/1.png, 3c/2.png, 4a/1.png, 4a/2.png ...
# already ran on my machine, so I don't have to worry about it
def batch_rename(root="/Users/prajjwaldangal/Documents/cs/summer2018/algo/downloads/"):
    # root : root directory that contains all the other hair types
    #       "/Users/prajjwaldangal/Documents/cs/summer2018/algo/downloads/"

    # rename image files into uniform numeric with ext
    for i in range(4):
        files = os.listdir(os.path.join(root,hair_types[i]))
        l = len(files)
        for j in range(l):
            # both src and dst should be absolute path
            os.rename(os.path.join(root, hair_types[i], files[j]),
                      os.path.join(root, hair_types[i], str(j + 1) + ".png"))
            # FileNotFoundError: [Errno 2]. No such file or directory: '495. 2v93z3l.jpg' -> '1.png'
    return

# 1. Run batch_rename once
# 2. Run load preprocess contours with the appropriate path.
originals, p, only_hair, conts_ls, canny = load_preprocess_contours(path, 4, "4b")
print(originals, p, only_hair, conts_ls, canny)

# returns image obtained after subtracting image from face_cascade coordinates
def face_cascade():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv2.imread('file1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # intensity image

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # faces is returned as a four number array/tuple, must be
    #       two coordinates that denote the centroid of the bounding
    #       rectangle and two values for height and width of the
    #       bounding rectangle.
    # It looks like: [[256 209 264 264]]
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #print(faces)
    x,y,w,h = faces[0]
    # set the face coordinates to o
    for i in range(y-h, y+h+1):
        for j in range(x-w, x+w+1):
            gray[i][j] = 0
    #gray[x-w:x+w+1][y-h:y+h+1] = 0
    #ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    #return gray - faces
    return gray

# subtracted_img = face_cascade()
# cv2.imshow('substracted_img', subtracted_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
