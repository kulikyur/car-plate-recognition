import cv2
import pytesseract
import numpy as np

# https://towardsdatascience.com/russian-car-plate-detection-with-opencv-and-tesseractocr-dce3d3f9ff5c
# https://github.com/AlexandrVP/Car-number-plate-detection-and-recognition/blob/master/car_number_detector_ssd.py
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"


image = cv2.imread('images/plate_10.jpg')

#cv2.imshow("Original", img)
#cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.bilateralFilter(gray, 11, 17, 17)

edged = cv2.Canny(gray, 170, 200)

cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

image1 = image.copy()
cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
#cv2.imshow("Canny", image1)
#cv2.waitKey(0)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCount = None

image2 = image.copy()
cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
#cv2.imshow("TOP 30 Contours", image2)
#cv2.waitKey(0)

count = 0
name = 1

for i in cnts:
    perimeter = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02*perimeter, True)

    if len(approx) == 4:
        NumberPlateCount = approx
        x, y, w, h = cv2.boundingRect(i)
        crp_img = image[y:y+h, x:x+w]
        cv2.imwrite(str(name)+ '.png', crp_img)
        name += 1

        break

cv2.drawContours(image, NumberPlateCount, -1, (0, 255, 0), 3)
#cv2.imshow("Final", image)
#cv2.waitKey(0)

crop_img_loc = '2.png'
#cv2.imshow("Cropped image", cv2.imread(crop_img_loc))
#cv2.waitKey(0)

#text = pytesseract.image_to_string(crop_img_loc, config = f'--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEHIKMNOPTXY0123456789')
#text_rus = pytesseract.image_to_string(crop_img_loc, lang='rus', config=tessdata_dir_config)



#print('Rus:', numbers)



# Convert image to grayscale
carplate_extract_img_gray = cv2.cvtColor(cv2.imread(crop_img_loc), cv2.COLOR_RGB2GRAY)
# Apply median blur
carplate_extract_img_gray_blur = cv2.medianBlur(carplate_extract_img_gray, 3) # kernel size 3

carplate_extract_img_gray_blur_edged = cv2.Canny(carplate_extract_img_gray_blur, 100, 100)
#cv2.imshow("Canny", carplate_extract_img_gray_blur_edged)
#cv2.waitKey(0)
"""
# Testing all PSM values
for i in range(3, 14):
    print(f'PSM: {i}')
    print(pytesseract.image_to_string(carplate_extract_img_gray_blur, config=f'--psm {i} --oem 3 -c tessedit_char_whitelist=ABCDEHKMNOPTXY0123456789'))
"""
#"""
for i in range(3, 14):
    print(f'PSM: {i}')
    read = pytesseract.image_to_string(carplate_extract_img_gray_blur, lang='rus', config=f'--psm {i}'
                                                           '--oem 0'
                                                           '-c tessedit_char_whitelist=0123456789АВСЕНКМРОТУХ'
                                                           '-c tessedit_char_blacklist=.,|><:;~`/][}{-=)($%^-!')
    read = ''.join(e for e in read if e.isalnum())
    if read[0] == '0':
        read = 'о' + read[1:]
    print(read)
#"""



