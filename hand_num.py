import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keyboard
import mediapipe as mp
preprocess_input=tf.keras.applications.efficientnet.preprocess_input
print('Let it go')
model=keras.models.load_model('hand_model.keras',custom_objects={"preprocess_input": preprocess_input})
label=['0','1','2','3','4','5','6','7','8','9']
hand_cascade = cv2.CascadeClassifier('hand.Cascade.1.xml')
# hand=mp.solutions.hands.Hands()

cap=cv2.VideoCapture(0)

if not cap.isOpened:
    print('cannot open the cam')

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame")
        break
#1-ci gelen shekili graysace edirik ki objecet dtection modelimiz eli teyin tsin
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hand = hand_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
    
#elin korditantalrin mueyyen edirik
    for (x, y, w, h) in hand:

        hand_color = frame[y:y+h, x:x+w]
#alnin a shekilin rengin ideyishirik ki mbizm model oyrensin
        hand_rgb = cv2.cvtColor(hand_color, cv2.COLOR_BGR2RGB)
        #resize edirk 
        hand_resized = cv2.resize(hand_rgb, (224, 224))
        #biz modeli oyrederken batch-batch oyredirkdi, burda ise bir bir geglir shekiller ona gore evvelden artirirq
        hand_input = np.expand_dims(hand_resized, axis=0)
        predictions = model.predict(hand_input)
        number_index = np.argmax(predictions[0])
        number = label[number_index]
        

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, number, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        if number_index==6:
                keyboard.press("volume mute")
    
        

cap.release()
cv2.destroyAllWindows()