# how they have exposed their hair
# change preferences python path to /usr/local/bin
import cv2
import algorithm_lib as lib

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


"""
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
