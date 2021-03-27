import cv2
import numpy as np
import pyttsx3

# Setup classifier

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

phone_cascade = cv2.CascadeClassifier('Phone_Cascade.xml')

rpalm_cascade = cv2.CascadeClassifier('rpalm.xml')
lpalm_cascade = cv2.CascadeClassifier('lpalm.xml')
fist_cascade = cv2.CascadeClassifier('fist.xml')
hand_cascade = cv2.CascadeClassifier('hand.xml')
right_cascade = cv2.CascadeClassifier('right.xml')
left_cascade = cv2.CascadeClassifier('left.xml')





class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.engine = pyttsx3.init()

    def __del__(self):
        # releasing camera
        self.cap.release()

    def get_frame(self):
        while True:
            ret, img = self.cap.read()



            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hand = hand_cascade.detectMultiScale(gray, 1.3, 5)
            fist = fist_cascade.detectMultiScale(gray, 1.3, 5)
            rpalm = rpalm_cascade.detectMultiScale(gray, 1.3, 5)
            lpalm = lpalm_cascade.detectMultiScale(gray, 1.3, 5)
            right = right_cascade.detectMultiScale(gray, 1.3, 5)
            left = left_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in hand:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

            for (x, y, w, h) in rpalm:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

            for (x, y, w, h) in lpalm:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

            for (x, y, w, h) in fist:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

            for (x, y, w, h) in right:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

            for (x, y, w, h) in left:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

            # Use classifier for detection
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            phones = phone_cascade.detectMultiScale(gray, 3, 9)

            if (len(faces) == 0):
                self.engine.say("Face can't be seen")
                self.engine.runAndWait()

            if (len(phones) > 0):
                self.engine.say("Phone is being used")
                self.engine.runAndWait()

            if ((len(hand) + len(rpalm) + len(lpalm) + len(fist) + len(right) + len(left)) == 0):
                self.engine.say("Hands are down")
                self.engine.runAndWait()

            for (x, y, w, h) in phones:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 10), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, 'Phone', (x - w, y - h), font, 0.5, (11, 255, 255), 2, cv2.LINE_AA)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                roi = img[y:y + h, x:x + w]

            cv2.imshow('img', img)

            if cv2.waitKey(60) & 0xFF == 27:
                print("SUCCESSFULLY COMPLETED")

            ret, jpeg = cv2.imencode('.jpg', img)
            return jpeg.tobytes()



