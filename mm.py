from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import pandas as pd
import webbrowser
import os
import streamlit as st

face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
classifier = load_model(r'model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

playlist_folder = 'Playlist'

def detect_emotion(frame):
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            emotion = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Save the detected emotion and retrieve the corresponding song links
            playlist_file = os.path.join(playlist_folder, emotion.lower() + '.csv')
            if os.path.isfile(playlist_file):
                playlist = pd.read_csv(playlist_file)
                if not playlist.empty:
                    song_row = playlist.sample(n=1)
                    song = song_row['Song'].values[0]
                    link = song_row['Link'].values[0]
                    webbrowser.open(link)
                    return emotion, song

    return None, None

def main():
    # Set up Streamlit layout
    st.title("Emotion-Based Music Player")
    st.subheader("Playing Song Based on Emotion")
    cap = None
    face_detected = False  # Variable to track if a face has been detected
    recapture = False  # Variable to track if recapture is requested
    emotion_detected = False  # Variable to track if an emotion has been detected
    #Last_emotion = None #variable to store the last emotion detected
    #Last_song = None #Variable to store the last song detected

    capture_button = st.button("Capture Emotion")
    recapture_button = st.button("Recapture Emotion")

    if capture_button:
        emotion_detected = False  # Reset the emotion detection flag
        cap = cv2.VideoCapture(0)
        for _ in range(10):
            _, frame = cap.read()
            emotion, song = detect_emotion(frame)
            if emotion:
                Last_emotion = emotion  #Storing the last emotion detected 
                Last_song = song  #Storing the last song played
                st.image(frame, channels="BGR")
                st.write(f"Detected Emotion: {emotion}")
                st.write("Playing Song:", song)
                emotion_detected = True
                break

            cv2.imshow('Emotion Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    if recapture_button and emotion_detected:
        main()
    st.subheader("Feedback")
    User_name = st.text_input("Your Name:")
    satisfaction_level = st.radio("Satisfaction Level", ["Satisfied", "Neutral", "Dissatisfied"])
    submit_button = st.button("Submit")

    if submit_button:
        # Store feedback in a DataFrame
        feedback_data = pd.DataFrame({
            "User Name": [User_name],
            #"Emotion": [Last_emotion],  
            #"Played Song": [Last_song],  
            "Satisfaction Level": [satisfaction_level]
        })
        # Check if the CSV file already exists
        if not os.path.isfile("Validation.csv"):
            feedback_data.to_csv("Validation.csv", index=False)  # Save the DataFrame as CSV
        else:
            feedback_data.to_csv("Validation.csv", mode="a", header=False, index=False)  # Append to existing CSV

        st.success("THANK YOU! \n Feedback submitted successfully!")

if __name__ == '__main__':
    main()
