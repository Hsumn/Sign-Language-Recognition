import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import tensorflow as tf
import numpy as np
import math
import time

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=2)
classifier = Classifier("/Users/hsumyatnoe/Desktop/Sign Language Detection/Model/keras_model.h5","/Users/hsumyatnoe/Desktop/Sign Language Detection/Model/labels.txt")

# Image size settings
offset = 20
imgSize = 300
labels = ["Hello", "How Are", "I Love You", "Meet", "Nice To", "No", "Please", "Thank You", "Yes", "You"]

while True:
    # Read a frame
    ret, img = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect hands
    hands, img = detector.findHands(img)  # img gets updated with drawn bbox

    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

            # Crop the hand region
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

                cv2.putText(img, labels[index], (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)

                cv2.imshow(f'ImageCrop_{hands.index(hand)}', imgCrop)
                cv2.imshow(f'ImageResized_{hands.index(hand)}', imgResize)

    # Show main image with detection
    cv2.imshow('Image', img)

    # Quit on 'q' press
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

from keras.preprocessing.image import ImageDataGenerator
model = tf.keras.models.load_model("/Users/hsumyatnoe/Desktop/Sign Language Detection/Model/keras_model.h5")
model.compile(optimizer=tf.keras.optimizers.legacy.Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])

data_dir = "/Users/hsumyatnoe/Desktop/Sign Language Detection/Data"
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(300, 300),
    batch_size=16,
    class_mode='categorical',
    shuffle=True
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=(300, 300),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

# Train and Save History
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator
)

loss, accuracy = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Predict and save
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

print("Saving new prediction and history data...")
np.save('history.npy', history.history)
np.save('y_true.npy', y_true)
np.save('y_pred.npy', y_pred)
