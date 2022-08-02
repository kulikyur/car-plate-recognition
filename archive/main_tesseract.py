# https://github.com/UB-Mannheim/tesseract/wiki - Tesseract installer for Windows
from PIL.Image import ImageTransformHandler
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")


def extract_num(img_filename):
    img = cv2.imread(img_filename)
    # Img To Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(10, 10))
    # crop portion
    for (x, y, w, h) in nplate:
        #wT, hT, cT = img.shape
        #a, b = int(wT), int(hT)
        a, b = (int(0.02*img.shape[0]), int(0.02*img.shape[1]))
        plate = img[y + a:y + h - a, x + b:x + w - b, :]
        # make the img more darker to identify LPR
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        (thresh, plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
        # read the text on the plate
        read = pytesseract.image_to_string(plate)
        print(read.upper())
        read = ''.join(e for e in read if e.isalnum())
        print(read.upper())

        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(img, (x - 1, y - 40), (x + w + 1, y), (51, 51, 255), -1)
        cv2.putText(img, read.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        #cv2.imshow("plate", plate)

    cv2.imwrite("../Result.png", img)
    cv2.imshow("Result", img)
    if cv2.waitKey(0) == 113:
        exit()
    cv2.destroyAllWindows()


extract_num("../images/plate_3.jpg")
