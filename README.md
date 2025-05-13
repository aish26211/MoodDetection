Mood Detection System

Overview
The Mood Detection System is a real-time facial emotion recognition project built using deep learning and OpenCV. It captures live video from the user's webcam, detects facial expressions, and classifies them into one of the predefined emotion categories using a trained Convolutional Neural Network (CNN).

Features:-
->Real-time webcam-based face detection

->Emotion classification using a CNN model

->Visual representation of detected emotions using a live-updating bar chart

->Simple and interactive UI using Streamlit

Emotion Categories
The system can detect and classify the following emotions:
->Angry
->Disgusted
->Fearful
->Happy
->Neutral
->Sad
->Surprised

Tech Stack
->Python
->TensorFlow / Keras
->OpenCV
->Streamlit
->Matplotlib
->NumPy / Pandas

Folder Structure:-

mood_detection_project/
├── src/
│   ├── model.h5                     # Trained CNN model weights
│   ├── haarcascade_frontalface_default.xml  # Face detection model
├── app.py                           # Main Streamlit application
├── requirements.txt                 # List of required Python libraries
└── README.md                        # Project documentation

How It Works:-
1.The application captures video through the webcam.

2.It uses a Haar Cascade classifier to detect faces in each frame.

3.Detected face regions are preprocessed (grayscale conversion, resizing, normalization).

4.The processed image is passed to a trained CNN model.

5.The model predicts the emotion category.

6.Results are displayed on-screen and summarized in a real-time bar chart.

Requirements:-
Python 3.7+

Streamlit

OpenCV

TensorFlow / Keras

NumPy

Matplotlib

Pandas

Ensure that your webcam is enabled and accessible by your browser/system.
