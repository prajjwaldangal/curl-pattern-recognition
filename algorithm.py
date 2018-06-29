# how they have exposed their hair
# change preferences python path to /usr/local/bin
import cv2
# import algorithm_lib as lib




######################################################################
# feature engineering: process of creating features from raw data


# linear  -->
######################  read stream from camera / webcam ###########
"""
cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(thresh, contours, -1, (0,255,0), 3)
    cv2.imshow("Key points of your hair", thresh)

    # intensity histogram
    cv2.waitKey(200)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

# rename file
def rename(dir_name="algo/downloads/3c hair"):
    files = os.listdir(path + dir_name)
    for i in range(len(files)):
        os.rename(files[i], str(i + 1) + ".png")


# make random chooser
def n_randm_imgs(n, N):
    imgs = []
    for i in range(n):
        r = random.randint(1, N)
        imgs.append(r)
    return imgs

"""

#########  face and eyes cascades  #######
"""
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('sachin.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# one way of segmenting using face cascade
face_coordinates = faces
only_hair = image - face_coordinates


"""


"""
Googleimagesdownload relevant code:
if arguments['proxy']:
            os.environ["http_proxy"] = arguments['proxy']
            os.environ["https_proxy"] = arguments['proxy']

"""


