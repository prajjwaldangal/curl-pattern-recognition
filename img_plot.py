# how they have exposed their hair
import cv2
import algorithm_lib as lib

cap = cv2.VideoCapture(0)

while (True):
	ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    img2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, 
			cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    cv2.imshow("Key points of your hair", frame)
    #cv2.imwrite(frame, )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()