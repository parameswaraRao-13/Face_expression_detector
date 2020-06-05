#import libraries
import os
import numpy
import cv2
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from PIL import Image
from keras.models import load_model
import cv2
import numpy as np
import time

mo = load_model('/home/ram/Desktop/facial_emotion_classification.h5')

mo.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

model = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'weights.caffemodel')

filename = "/home/ram/Desktop/submissions/R22MG6.jpg"
image = cv2.imread(filename)
(h, w) = image.shape[:2]

# get our blob which is our input image
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                               (103.93, 116.77, 123.68))  # input the blob into the model and get back the detections
model.setInput(blob)
detections = model.forward()
for i in range(0, detections.shape[2]):
    box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])

    (startX, startY, endX, endY) = box.astype("int")
    confidence = detections[0, 0, i, 2]
    if (confidence > 0.96):
        # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 1)
        try:
            iii = Image.open(filename)
            crop_image = iii.crop((startX-200, startY-200, endX+200, endY+200))
            new_img = crop_image.resize((150, 150))
            m = load_model('/home/ram/Desktop/facial_emotion_classification.h5')
            m.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
            new_img = np.reshape(new_img, [1, 150, 150, 3])
            classes = m.predict_classes(new_img)
            #print(classes[0])
            i = int(classes)
            if i == 0:
                cv2.rectangle(image, (startX-30, startY-30), (endX+30, endY+30), (0, 255, 0), 2)
                s = "anger"
                print("person has anger face expression")
                text = "{:.2f}%".format(confidence * 100)
                cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(image, s, (startX, endY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            elif i == 1:
                cv2.rectangle(image, (startX-30, startY-30), (endX+30, endY+30), (0, 255, 0), 2)
                s = "disgust"
                print("person has disgust face expression")
                text = "{:.2f}%".format(confidence * 100)
                cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(image, s, (startX, endY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            elif i == 2:
                cv2.rectangle(image, (startX - 30, startY - 30), (endX + 30, endY + 30), (0, 255, 0), 2)
                s = "happy"
                print("person has happy face expression")
                text = "{:.2f}%".format(confidence * 100)
                cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(image, s, (startX, endY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            elif i == 3:
                cv2.rectangle(image, (startX-30, startY-30), (endX+30, endY+30), (0, 255, 0), 2)
                s = "neutral"
                print("person has neutral face expression")
                text = "{:.2f}%".format(confidence * 100)
                cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(image, s, (startX, endY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            elif i == 4:
                cv2.rectangle(image, (startX-30, startY-30), (endX+30, endY+30), (0, 255, 0), 2)
                s = "sad"
                print("person has sad face expression")
                text = "{:.2f}%".format(confidence * 100)
                cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(image, s, (startX, endY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            elif i == 5:
                cv2.rectangle(image, (startX-30, startY-30), (endX+30, endY+30), (0, 255, 0), 2)
                s = "surprise"
                print("person has surprise face expression")
                text = "{:.2f}%".format(confidence * 100)
                cv2.putText(image, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(image, s, (startX, endY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        except:
            print("unable to crop")

imshow('face expression detection', image)
if cv2.getWindowProperty('face detection', cv2.WND_PROP_VISIBLE) == -1:
    cv2.destroyAllWindows()

else:
    waitKey(0)
    # close the window
    cv2.destroyAllWindows()
cv2.destroyAllWindows()