import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import tensorflow as tf
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("/Users/hsumyatnoe/Desktop/Sign Language Detection/Model/keras_model.h5","/Users/hsumyatnoe/Desktop/Sign Language Detection/Model/labels.txt")

offset = 20
imgSize = 300
labels = ["Hello", "How Are", "I Love You", "Meet", "Nice To", "No", "Please", "Thank You", "Yes", "You"]

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    hands, img = detector.findHands(img)  

    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

            imgCrop = img[max(0, y - offset): min(y + h + offset, img.shape[0]), 
                          max(0, x - offset): min(x + w + offset, img.shape[1])]

            if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    newWidth = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (newWidth, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize-newWidth)/2)
                    imgWhite[:, wGap: newWidth + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite)
                    print(prediction,index)

                else:
                    k = imgSize / w
                    newHeight = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, newHeight))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - newHeight) / 2)
                    imgWhite[hGap: newHeight + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite)
 
                label = labels[index]
                confidence = prediction[index] * 100  

                text = f'{label} ({confidence:.2f}%)'
                cv2.putText(img, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)

                cv2.imshow(f'ImageCrop_{hands.index(hand)}', imgCrop)
                cv2.imshow(f'ImageResized_{hands.index(hand)}', imgResize)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
