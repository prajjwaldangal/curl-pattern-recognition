import sys
import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt
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
parser.add_argument('-i', '--input-dir', default='/Users/prajjwaldangal/Documents/cs/summer2018/algo/data/',
                    help='Directory containing images (png format) to process')
# parser.add_argument()
args = parser.parse_args()


class InputSpec(object):

    def __init__(self, path=args.input_dir, BATCH__SIZE=10):
        self._path = path
        self._BATCH_SIZE = BATCH__SIZE
        # self._n =

def dots(n):
    i = n % 10
    # dots on the terminal to show busy process
    sys.stdout.write("." * i)
    sys.stdout.flush()# flush: write to terminal without waiting for the buffer to fill up


# load images into the program
def load_preprocess_contours(hair_type, n, resize_dim=None, extra="train", segmented=True, inv=True,
                             get_conts=False, get_canny=False):
    """
    args:
		hair_type : the hair type
		n : number of image files
		extra: 'train' or 'test'
		segmented: where to look for files (segmented or unsegmented dir inside hair_type dir)
		inv : whether or not to invert the binary image, for our algorithm we want it to be True

	:rtype: list of binary images, original images, grays as well as
			contour lists and canny edges
	"""

    # TO-DO: add slider for threshold value
    root = args.input_dir
    j = hair_types.index(hair_type.strip().lower())
    originals = []
    grays = []
    only_hair = []
    conts_ls = []
    canny = []
    print("Processing images of type {}.......".format(hair_type))
    for i in range(n):
        if segmented:
            path = os.path.join(root, extra, hair_types[j], "segmented", str(i + 1) + '.png')
        else:
            path = os.path.join(root, extra, hair_types[j], "unsegmented", str(i + 1) + '.png')
        # print for output
        print("Loading {}".format(path.split("segmented")[1][1:]))
        print(path)
        img = cv.imread(path)
        if resize_dim:
            img = cv.resize(img, resize_dim)
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        dots(i)
        # convert to color intensity image to binary image
        gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # gray_image = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        if inv:
            ret, thresh = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV)
        else:
            ret, thresh = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY)
        # conts_ls is a list of contoured images
        # cv.CHAIN_APPROX_SIMPLE requires only 4pts to represent a square contour vs
        #	cv.CHAIN_APPROX_NONE which would require thousands of points
        if get_conts:
            img2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,
                                                    cv.CHAIN_APPROX_SIMPLE)
            conts_ls.append(contours)

        # [1 0 -1]                          [1   1  1]
        # [1 0 -1]   for vertical edges,    [0   0  0]  should be for horizontal edges
        # [1 0 -1]                          [-1 -1 -1]
        # gives you nice, pointful edges, called L2 gradient
        if get_canny:
            edges = cv.Canny(thresh, threshold1=100, threshold2=255, L2gradient=True)
            canny.append(edges)

        originals.append(img)
        grays.append(gray_image)
        only_hair.append(thresh)
        # Carriage return ("\r") means to return to the beginning of the current line without advancing downward.
        # The name comes from a printer's carriage
        sys.stdout.write("\r")
    dots(7)
    sys.stdout.write("\n")
    return only_hair, grays, originals, conts_ls, canny


# returns image obtained after subtracting image from face_cascade coordinates
def face_cascade():
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv.imread('file1.jpg')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # intensity image

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # faces is returned as a four number array/tuple, must be
    #       two coordinates that denote the centroid of the bounding
    #       rectangle and two values for height and width of the
    #       bounding rectangle.
    # It looks like: [[256 209 264 264]]
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # print(faces)
    x, y, w, h = faces[0]
    # set the face coordinates to o
    for i in range(y - h, y + h + 1):
        for j in range(x - w, x + w + 1):
            gray[i][j] = 0
    # gray[x-w:x+w+1][y-h:y+h+1] = 0
    # ret, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)

    # return gray - faces
    return gray


# subtracted_img = face_cascade()
# cv.imshow('substracted_img', subtracted_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# plots from a list consisting
def plotting(ls, fig_num, dim):
    """
    :param ls: list of images
    :param fig_num: figure number
    :param dim: 2-ple for no. of rows and cols in the plot
    :return: NA
    """
    cap = dim[0] * dim[1]
    if len(ls) > cap:
        print("Last {} discarded".format(abs(cap-len(ls))))
    elif len(ls) < cap:
        print("Adjusting axes")
        dots(4)
        cap = len(ls)

    plt.figure(fig_num)
    # len(ls) = 18
    # cap = 4*4 = 16
    for i in range(cap):
        plt.subplot(3, int(cap/3) + 1, i+1)
        plt.imshow(ls[i])

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


# Specifically, compare original, contours and canny images.
# TO-DO: add hair_type argument and changes
# Special case of plotting function
def plotting2(hair_type, n, extra="train", segmented=True):
    if segmented:
        only_hair, grays, originals, conts_ls, canny = load_preprocess_contours(hair_type.strip().lower(), n)
    else:
        only_hair, grays, originals, conts_ls, canny = load_preprocess_contours(hair_type.strip().lower(), n,
                                                                                segmented=False)
    conts_imgs = []
    for i in range(n):
        # myFile = "{Path}{dir_ini}{ind}{form}".format(Path=path, dir_ini="figures/cont_",
        #                                              ind=i + 1, form='.png')
        contrs = cv.drawContours(only_hair[i], conts_ls[i], -1, (0, 255, 0), 3)
        conts_imgs.append(contrs)
    plotting(only_hair, 1, (5,5))
    # try plotting(conts_ls, ..)
    plotting(conts_imgs, 2, (5,5))
    plotting(canny, 3, (5,5))

    return


# run this function once to rename image files
# into numeric filenames like, 3c/1.png, 3c/2.png, 4a/1.png, 4a/2.png ...
# already ran on my machine, so I don't have to worry about it
def batch_rename(hair_dir, extra="train", segmented=True):
    """

    :param hair_dir: hair type (3c, 4a ...)
    :param extra: train or test
    :param segmented: rename segmented dir or unsegmented dir
    :return: NA
    """
    # rename image files into uniform numeric with exts
    if segmented:
        path = os.path.join(args.input_dir, extra, hair_dir, "segmented")
    else:
        path = os.path.join(args.input_dir, extra, hair_dir, "unsegmented")
    filenames = os.listdir(path)
    for idx, filename in enumerate(filenames):
        os.rename(os.path.join(path, filename), os.path.join(path, str(idx + 1) + ".png"))


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
    r, h, c, w = 250, 90, 400, 125  # simply hardcoded the values
    track_window = (c, r, w, h)
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
    term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    _, _, imgs, _, _ = load_preprocess_contours(hair_type, n, (50,50), segmented=False)
    for i in range(n):
        img = imgs[i]
        roi = img[r:r+h, c:c+w]
        hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
        # mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
        roi_hist = cv.calcHist([hsv_roi],[0],None,[180],[0,180])
        cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        ret, track_window = cv.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv.rectangle(img, (x,y), (x+w,y+h), 255,2)
        cv.imshow('img2',img2)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # applyColorMap applies a GNU Octave/MATLAB equivalent colormap on a given image
        # cv.applyColorMap(src, colormap [, dst]) -> dst
        # ret, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)

        # apply cv.calcHist and cv.calcBackProject on thresh first of all
        """
            ######### Model histogram and meanshift  #############
            s = [[1,2,3],[1,2,3],[1,2,3]]
            calcHist(s) returns  [3,3,3]
            calcBackProject(s, hist, 2) returns  [[6,6,6],[6,6,6],[6,6,6]]
            hist = cv2.calcHist([s], [0], None, [3], [1,4])
            dst = cv2.calcBackProject([s], [0], hist, [1,4], 2)
        """

    return imgs


# imgs = mean_shift("3c", 100)
# print("Length of imgs is: {}".format(len(imgs)))
# plt2.Index([imgs], ["3c"]).plot_batch()

# if __name__ == '__main__':
    # batch_rename("4c", extra="train", segmented=False)
    # mean_shift("4c", 10)


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

# flixify start from 30, 32, 
###################### Mean shift segmentation method #####################
Histograms are collected counts of data organized into a set of predefined bins
intensity values (or ranges of intensity values)  on x axis, frequency on y axis for a given (a x b) region of img

Monday meeting issues to raise: add as collaborators on github, somebody needs to look into k-means clustering for 
segmentation."""
