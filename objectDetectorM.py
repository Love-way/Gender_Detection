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
maskModel = tensorflow.keras.models.load_model('as_model.h5')
model = tensorflow.keras.models.load_model('KJCmodel.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# set the camera
cam = cv2.VideoCapture(1)

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
    maskprediction = maskModel.predict(data)
    prediction = model.predict(data)
    # print(prediction)

    for m, wm, bg, bk, bt, ct, ch, cp, ctn, dg, dr, fd, gl, ht, ot, pn, ph, pl, sf, tb, t, tc, tr, tv, vl, ws,wh, wn in prediction:

        pr = max(m, wm, bg, bk, bt, ct, ch, cp, ct, dg, dr, fd, gl, ht, ot, pn, ph, pl, sf, tb, t, tc, tr, tv, vl, ws,wh, wn)
        print(pr)

        if pr >= 0.5:
            if pr == m:
                text = 'Hi, Sir'
                pass
            elif pr == wm:
                text = 'Hello, Madam'
                pass
            elif pr == bg:
                text = 'bag'
                pass
            elif pr == bk:
                text = 'book'
                pass
            elif pr == bt:
                text = 'bottle'
                pass
            elif pr == ct:
                text = 'cat'
                pass
            elif pr == ch:
                text = 'chair'
                pass
            elif pr == cp:
                text = 'computer'
                pass
            elif pr == ctn:
                text = 'container'
                pass
            elif pr == dg:
                text = 'dog'
                pass
            elif pr == dr:
                text = 'door'
                pass
            elif pr == fd:
                text = 'food'
                pass
            elif pr == gl:
                text = 'glasses'
                pass
            elif pr == ht:
                text = 'hat'
                pass
            elif pr == ot:
                text = 'outlet'
                pass
            elif pr == pn:
                text = 'pen'
                pass
            elif pr == ph:
                text = 'phone'
                pass
            elif pr == pl:
                text = 'pole'
                pass
            elif pr == sf:
                text = 'sofa'
                pass
            elif pr == tb:
                text = 'table'
                pass
            elif pr == t:
                text = 'trash'
                pass
            elif pr == tc:
                text = 'trash can'
                pass
            elif pr == tr:
                text = 'tree'
                pass
            elif pr == tv:
                text = 'tv'
                pass
            elif pr == vl:
                text = 'vehicle'
                pass
            elif pr == ws:
                text = 'wall switch'
                pass
            elif pr == wh:
                text = 'watch'
                pass
            elif pr == wn:
                text = 'window'
                pass
        else:
            text=""
            pass
        
        # img = cv2.resize(img, (500, 500))
        cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0), 2)
        
    cv2.imshow('KJC Mask Scanner', frame)
    speak(text)
    exitKey = cv2.waitKey(1)
    if exitKey == ord('q') or exitKey == ord('Q'):
        break

    # time.sleep(2)

cam.release()
cv2.destroyAllWindows()
os.remove('scan.jpg')


