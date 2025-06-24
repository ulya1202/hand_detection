#IMPORT
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keyboard
import mediapipe as mp
import tkinter as tk
from PIL import Image
#models and preprocesses
##variables

preprocess_input=tf.keras.applications.efficientnet.preprocess_input
label=['0','1','2','3','4','5','6','7','8','9']

model=keras.models.load_model('hand_model.keras',custom_objects={"preprocess_input": preprocess_input})
mp_hand= mp.solutions.hands
hands=mp_hand.Hands(max_num_hands=1,min_detection_confidence=0.5)
# img = cv2.imread('images (5).jpg')
# image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# result=hands.process(image)
# landmarks=result.multi_hand_landmarks
###CAM______________________
cap=cv2.VideoCapture(0)

if not cap.isOpened:
    print('cannot open the cam')

while True:
    ret, img = cap.read()

    if not ret:
        print("Can't receive frame")
        break
    ##______________
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result=hands.process(image)
    landmarks=result.multi_hand_landmarks
##START____________________________________
    xl=[]
    yl=[]
    if landmarks:
        for landmark in landmarks:
            for i , l in enumerate(landmark.landmark):
                h,w,d=img.shape
                xl.append(int(l.x*w))
                yl.append(int(l.y*h))

        xmin,ymin,xmax,ymax=min(xl), min(yl),max(xl),max(yl)
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        new_hand = image[ymin:ymax, xmin:xmax]

        hand_resized = cv2.resize(new_hand, (224, 224))
        hand_input = np.expand_dims(hand_resized, axis=0)
        predictions = model.predict(hand_input)
        number_index = np.argmax(predictions[0])
        number = label[number_index]
        if number_index==0:
            keyboard.send("volume mute")
        if number_index==3:
            keyboard.send("volume up")
        # if number_index==2:
        #     keyboard.send('volume low')
        # if number_index==5:
        #     keyboard.send("brightness up")
        # if number_index==3:
        #     keyboard.send("brightness down") 
       

        cv2.rectangle(img, (xmin - 20, ymin - 20),(xmax + 20, ymax + 20),
                                    (0, 255 , 0) , 2)
        cv2.putText(img, number, (xmax, ymax),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Rectangle Example", img)
         
        
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break                      # Pəncərəni gözlətmək üçün
        
cv2.destroyAllWindows()  





