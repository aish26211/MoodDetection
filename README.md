# 🧠 Mood Detection System

## 📌 Overview
The **Mood Detection System** is a real-time facial emotion recognition project built using **Deep Learning** and **OpenCV**. It captures live video from the user's webcam, detects facial expressions, and classifies them into one of seven predefined emotion categories using a trained **Convolutional Neural Network (CNN)**.

---

## ✨ Features

- Real-time webcam-based face detection  
- Emotion classification using a trained CNN model  
- Live-updating bar chart visualization of detected emotions  
- Simple, responsive, and interactive UI powered by **Streamlit**

---

## 😊 Emotion Categories

The system can detect and classify the following emotions:

- Angry  
- Disgusted  
- Fearful  
- Happy  
- Neutral  
- Sad  
- Surprised

---

## 🛠 Tech Stack

- **Python**  
- **TensorFlow / Keras**  
- **OpenCV**  
- **Streamlit**  
- **Matplotlib**  
- **NumPy**  
- **Pandas**

---

## 📁 Folder Structure

```bash
mood_detection_project/
├── src/
│   ├── model.h5                          # Trained CNN model weights
│   ├── haarcascade_frontalface_default.xml  # Haar Cascade face detection model
├── app.py                                # Main Streamlit application for mood detection
├── requirements.txt                      # List of required Python libraries
└── README.md                             # Project documentation

```
---

## ⚙️ How It Works

1. The application captures video through the webcam.  
2. It uses a Haar Cascade classifier to detect faces in real-time.  
3. Detected face regions are converted to grayscale, resized to 48x48, and normalized.  
4. The processed image is passed to a pretrained CNN model.  
5. The model outputs the predicted emotion label.  
6. Predictions are displayed on the screen and visualized via a live bar chart.

---

## 📚 References

The following datasets were used to train and evaluate the emotion detection model:

- **FER-2013 Emotion Recognition Dataset**  
  Source: [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)  
  Description: Contains grayscale 48x48 pixel face images categorized into 7 emotion classes: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.

- **Haar Cascade for Face Detection**  
  Source: [https://github.com/opencv/opencv/tree/master/data/haarcascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)  
  Description: Pretrained XML model for detecting human frontal faces in images and video frames.

---

## 📦 Requirements

Make sure the following libraries are installed:

- Python 3.7+  
- Streamlit  
- OpenCV  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Pandas

You can install all dependencies using:

```bash
pip install -r requirements.txt
