# By  KJC

# with the speak function

import tensorflow.keras
import numpy as np
import cv2
import pyttsx3
import math
import time
import os


# set the speaker parameters
speaker = pyttsx3.init()
voices = speaker.getProperty('voices')
speaker.setProperty('voice', voices[1].id)
speaker.setProperty('rate', 200)


# set a speak function
def speak(arg):
    speaker.say(arg)
    speaker.runAndWait()

# disable scientific notation for clarity
np.set_printoptions(suppress=False)

# load the model
model = tensorflow.keras.models.load_model('gendermask.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# set the camera
cam = cv2.VideoCapture(0)

while cam.isOpened():

    # setting the image framework
    check, frame = cam.read()

    cv2.imwrite('scan.jpg', frame)

    img = cv2.imread('scan.jpg')
    img = cv2.resize(img, (224, 224))

    imageArray = np.asarray(img)

    # normalize the image
    normalizedImage = (imageArray.astype(np.float32)/127.0) - 1
    data[0] = normalizedImage

    # prediction
    prediction = model.predict(data)
    # print(prediction)

    for m, mm, w, wm, e in prediction:

        pr = max(m, mm, w, wm, e)
        # print(pr)

        if pr == m :
            text = 'Man'
            this = 'Please Sir wear your mask!'
            text_color = (0,0,255)

        elif pr == mm:
            text = 'Man'
            this = 'Welcome in Gentleman'
            text_color = (0,255,0)

        elif pr == w:
            text = 'Woman'
            this = 'Please Madame wear your mask!'
            text_color = (0,0,255)

        elif pr == wm:
            text = 'Woman'
            this = 'Welcome in Lady'
            text_color = (0,255,0)

        elif pr == e:
            text = ''
            this = ''
            text_color = (0,255,0)

        
        # img = cv2.resize(img, (500, 500))
        cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_COMPLEX,1,text_color, 2)
        
    cv2.imshow('KJC Mask Scanner', frame)
    speak(this)
    exitKey = cv2.waitKey(1)
    if exitKey == ord('q') or exitKey == ord('Q'):
        break

    # time.sleep(2)

cam.release()
cv2.destroyAllWindows()
os.remove('scan.jpg')


