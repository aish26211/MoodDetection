import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sqlite3
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import streamlit as st

# Database setup
def init_db():
    conn = sqlite3.connect("emotion_data.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    name TEXT,
                    password TEXT
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS emotions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    student_name TEXT,
                    timestamp TEXT,
                    emotion TEXT
                )''')
    conn.commit()
    conn.close()

init_db()

# Emotion Labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Initialize session state for login and detection
temp_login = None
if 'student_id' not in st.session_state:
    st.session_state.student_id = None
if 'student_name' not in st.session_state:
    st.session_state.student_name = None

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("src/haarcascade_frontalface_default.xml")

# Build & Load Model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    model.load_weights("src/model.h5")
    return model

# Load Model
try:
    model = build_model()
except Exception as e:
    st.error(f"Error Loading Model: {e}")

# Login Page
st.sidebar.header("User Login")
login_id = st.sidebar.text_input("Student ID")
login_name = st.sidebar.text_input("Student Name")
login_button = st.sidebar.button("Login")

if login_button:
    if login_id and login_name:
        st.session_state.student_id = login_id
        st.session_state.student_name = login_name
        st.success(f"Logged in as {login_name}")
    else:
        st.error("Please enter both Student ID and Name")

if not st.session_state.student_id:
    st.warning("Please log in to detect emotions.")
    st.stop()

# Streamlit UI
st.title("Emotion Detection & Analysis Dashboard")
analysis_type = st.sidebar.radio("Select Mode", ["Live Detection", "Past Analysis", "Admin Dashboard"])

def save_emotion(student_id, student_name, emotion):
    conn = sqlite3.connect("emotion_data.db")
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO emotions (student_id, student_name, timestamp, emotion) VALUES (?, ?, ?, ?)",
              (student_id, student_name, timestamp, emotion))
    conn.commit()
    conn.close()

def get_past_data():
    conn = sqlite3.connect("emotion_data.db")
    df = pd.read_sql("SELECT * FROM emotions", conn)
    conn.close()
    return df

# Live Detection
if analysis_type == "Live Detection":
    st.subheader("Enable Camera for Detection")
    detect_button = st.button("Detect Emotion")
    stop_button = st.button("Stop Camera")
    
    if detect_button:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=-1) / 255.0  
                predictions = model.predict(face)
                detected_emotion = emotion_dict[np.argmax(predictions)]
                save_emotion(st.session_state.student_id, st.session_state.student_name, detected_emotion)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            stframe.image(frame, channels="BGR", use_column_width=True)
    
    if stop_button:
        cap.release()

# Past Analysis
elif analysis_type == "Past Analysis":
    st.subheader("Past Data Analysis")
    df = get_past_data()
    if not df.empty:
        st.dataframe(df)
        st.subheader("Emotion Distribution")
        fig, ax = plt.subplots()
        emotion_counts = df["emotion"].value_counts()
        ax.pie(emotion_counts, labels=emotion_counts.index, autopct="%1.1f%%", colors=['red', 'green', 'blue', 'yellow', 'gray', 'purple', 'orange'])
        ax.set_title("Past Emotion Distribution")
        st.pyplot(fig)
    else:
        st.info("No past data available.")

# Admin Dashboard
elif analysis_type == "Admin Dashboard":
    st.subheader("Admin Panel - Weekly Reports")
    df = get_past_data()
    if not df.empty:
        sad_students = df[df['emotion'].isin(["Sad", "Frustrated"])]
        flagged_students = sad_students.groupby(['student_id', 'student_name']).count().reset_index()
        st.write("Flagged students for therapy sessions:")
        st.dataframe(flagged_students)
    else:
        st.info("No data available.")
