import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time 

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300
counter = 0
folder = "/Users/hsumyatnoe/Desktop/Sign Language Detection/Data/You"

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    hands, img = detector.findHands(img)  

    if hands:
        for hand in hands:
          if 'bbox' in hand: 
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
                    wGap = math.ceil((imgSize-newWidth)/2)
                    imgWhite[:, wGap: newWidth + wGap] = imgResize

                else:
                    k = imgSize / w
                    newHeight = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, newHeight))
                    hGap = math.ceil((imgSize - newHeight) / 2)
                    imgWhite[hGap: newHeight + hGap, :] = imgResize

                cv2.imshow(f'ImageCrop_{hands.index(hand)}', imgCrop)
                cv2.imshow(f'ImageResized_{hands.index(hand)}', imgResize)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        filename = f"{folder}/Image_{time.time()}.jpg"
        cv2.imwrite(filename, imgWhite)
        print(counter)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
