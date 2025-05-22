import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("/Users/hsumyatnoe/Desktop/Sign Language Detection/Model/keras_model.h5")

testdata = ImageDataGenerator(rescale=1./255)
testgen = testdata.flow_from_directory(
    "/Users/hsumyatnoe/Desktop/Sign Language Detection/Data",
    target_size=(300, 300),
    batch_size=16,
    class_mode='categorical',
    shuffle=False
)

y_pred_probs = model.predict(testgen)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = testgen.classes
class_labels = list(testgen.class_indices.keys())

# Classification report
print("Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()