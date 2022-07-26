import cv2

#############################################
frameWidth = 640
frameHeight = 480
nPlateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea = 200
color = (255, 0, 255)
x1, y1, y2, x2, = 655, 420,  480,  430
###############################################
img = cv2.imread('images/plate_3.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
numberPlates = nPlateCascade.detectMultiScale(imgGray, scaleFactor=1.1, minNeighbors=11, minSize=(20, 20))

for (x, y, w, h) in numberPlates:
    area = w * h
    if area > minArea:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, "Number Plate", (x, y - 5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
        imgRoi = img[y:y + h, x:x + w]
        #cv2.imshow("ROI", imgRoi)

cv2.imshow("Result", img)
cv2.waitKey(0)
###########################################################


###########################################################
# cv2.rectangle(img, (x1, x2), (y1, y2), (0, 255, 0), 2)
# cv2.putText(img, "Number Plate", (x1, x2 - 5),
#             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
# cv2.imshow("Result", img)
# cv2.waitKey(0)
###########################################################
