import cv2
import numpy as np
import pytesseract
import platform

if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

frameWidth = 640    #Frame Width
franeHeight = 480   # Frame Height

plateCascade = cv2.CascadeClassifier('libs/haarcascade_russian_plate_number.xml')
minArea = 500

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, franeHeight)
cap.set(10, 150)
count = 0

while True:
    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = plateCascade .detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #cv2.putText(img, "NumberPlate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            imgRoi = img[y:y+h, x:x+w]
            #extract_img_gray = cv2.cvtColor(cv2.imread(imgRoi), cv2.COLOR_RGB2GRAY)
            #extract_img_gray_blur = cv2.medianBlur(extract_img_gray, 3)
            #text = pytesseract.image_to_string(extract_img_gray_blur, config=f'--psm 13 --oem 3 -c tessedit_char_whitelist=ABCDEHIKMNOPTXY0123456789')
            text = pytesseract.image_to_string(imgRoi, lang='rus', config=f'--psm 12'
                                                                                                  '--oem 0'
                                                                                                  '-c tessedit_char_whitelist=0123456789АВСЕНКМРОТУХ'
                                                                                                  '-c tessedit_char_blacklist=.,|><:;~`/][}{-=)($%^-!')
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("ROI", imgRoi)
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

