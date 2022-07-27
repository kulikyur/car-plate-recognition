import cv2
import numpy as np

#############################################
nPlateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
minArea = 50
color = (255, 0, 255)
###############################################

# Создаем объект захвата видео, в этом случае мы читаем видео из файла
vid_capture = cv2.VideoCapture('images/IMG_4163.mp4')
object_detector = cv2.createBackgroundSubtractorMOG2()

while vid_capture.isOpened():
    # Метод vid_capture.read() возвращают кортеж, первым элементом является логическое значение
    # а вторым кадр
    ret, frame = vid_capture.read()

    mask = object_detector.apply(frame)
    numberPlates = nPlateCascade.detectMultiScale(mask, scaleFactor=1.1, minNeighbors=11, minSize=(20, 20))

    for x, y, w, h in numberPlates:
        print(x, y, w, h)
        area = w * h
        if area > minArea:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(frame, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

    if ret == True:
        cv2.imshow('Look', frame)
        key = cv2.waitKey(30)

        if (key == ord('q')) or key == 27:
            break
    else:
        break
# Освободить объект захвата видео
vid_capture.release()
cv2.destroyAllWindows()
