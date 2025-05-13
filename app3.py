import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import sqlite3
import face_recognition
import uuid
import os

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
emotion_counts = {emotion: 0 for emotion in emotion_dict.values()}


def init_db():
    conn = sqlite3.connect("emotion_data.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS EmotionData (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    emotion TEXT NOT NULL,
                    count INTEGER DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source TEXT,
                    session_id TEXT
                 )''')
    c.execute('''CREATE TABLE IF NOT EXISTS FaceRegistry (
                    customer_id TEXT PRIMARY KEY,
                    face_encoding BLOB NOT NULL,
                    customer_name TEXT
                 )''')
    conn.commit()
    conn.close()

def store_emotion(customer_id, emotion, count, source, session_id=None):
    conn = sqlite3.connect("emotion_data.db")
    c = conn.cursor()
    c.execute("INSERT INTO EmotionData (customer_id, emotion, count, source, session_id) VALUES (?, ?, ?, ?, ?)",
              (customer_id, emotion, count, source, session_id))
    conn.commit()
    conn.close()

def store_face_encoding(customer_id, encoding, customer_name=None):
    conn = sqlite3.connect("emotion_data.db")
    c = conn.cursor()
    encoding_blob = encoding.tobytes()
    c.execute("INSERT OR REPLACE INTO FaceRegistry (customer_id, face_encoding, customer_name) VALUES (?, ?, ?)",
              (customer_id, encoding_blob, customer_name))
    conn.commit()
    conn.close()

def load_known_faces():
    conn = sqlite3.connect("emotion_data.db")
    c = conn.cursor()
    c.execute("SELECT customer_id, face_encoding, customer_name FROM FaceRegistry")
    rows = c.fetchall()
    known_face_encodings = []
    known_customer_ids = []
    known_customer_names = []
    for customer_id, encoding_blob, customer_name in rows:
        encoding = np.frombuffer(encoding_blob, dtype=np.float64)
        known_face_encodings.append(encoding)
        known_customer_ids.append(customer_id)
        known_customer_names.append(customer_name if customer_name else "Unnamed")
    conn.close()
    return known_face_encodings, known_customer_ids, known_customer_names

def identify_face(frame, known_encodings, known_ids, known_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        return None, None
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    if not face_encodings:
        return None, None
    
    face_encoding = face_encodings[0]  
    if known_encodings:
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if face_distances[best_match_index] < 0.5:  
            return known_ids[best_match_index], known_names[best_match_index]
    
    new_customer_id = str(uuid.uuid4())[:8]
    customer_name = st.session_state.get(f"name_{new_customer_id}", None)
    if customer_name is None:
        with st.form(key=f"form_{new_customer_id}"):
            customer_name_input = st.text_input(f"New face detected (ID: {new_customer_id})! Please enter your name:")
            submit_button = st.form_submit_button(label="Submit Name")
            if submit_button and customer_name_input:
                customer_name = customer_name_input
                st.session_state[f"name_{new_customer_id}"] = customer_name
                store_face_encoding(new_customer_id, face_encoding, customer_name)
                
                global known_face_encodings, known_customer_ids, known_customer_names
                known_face_encodings, known_customer_ids, known_customer_names = load_known_faces()
    return new_customer_id, customer_name


def load_past_data(customer_id=None):
    conn = sqlite3.connect("emotion_data.db")
    query = "SELECT customer_id, emotion, count, timestamp, source, session_id FROM EmotionData"
    if customer_id:
        query += " WHERE customer_id = ?"
        df = pd.read_sql_query(query, conn, params=(customer_id,))
    else:
        df = pd.read_sql_query(query, conn)
    conn.close()
    return df


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
    model_weights_path = "src/model.h5"
    
    if not os.path.exists(model_weights_path):
        st.error(f"Model weights file '{model_weights_path}' not found. Please ensure it exists in the working directory.")
        st.stop()
    try:
        model.load_weights(model_weights_path)
    except Exception as e:
        st.error(f"Failed to load model weights from '{model_weights_path}': {str(e)}")
        st.stop()
    return model

face_cascade = cv2.CascadeClassifier("src/haarcascade_frontalface_default.xml")
if face_cascade.empty():
    st.error("Error: Could not load haarcascade_frontalface_default.xml. Check the file path.")
    st.stop()


st.title("Emotion Detection & Analysis Dashboard")
init_db()
st.subheader("Analyze emotions in real-time or from past data.")


st.sidebar.header("Analysis Options")
manual_customer_id = st.sidebar.text_input("Override Customer ID (optional)", "")
analysis_type = st.sidebar.radio("Select Mode", ["Live Detection", "Past Analysis"])
st.sidebar.markdown("---")
customer_id_display = st.sidebar.empty()  


try:
    model = build_model()
except Exception as e:
    st.error(f"Error initializing model: {e}")
    st.stop()


known_face_encodings, known_customer_ids, known_customer_names = load_known_faces()

if analysis_type == "Live Detection":
    st.subheader("Enable Camera or Upload Image for Detection")
    option = st.radio("Select Input Method", ["Use Camera", "Upload Image"])
    run = st.button("Start Webcam")
    stop = st.button("Stop Webcam")

    stats_chart = st.sidebar.empty()
    stframe = st.empty()
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    session_id = "SESSION_" + str(np.random.randint(1000, 9999))

    cap = None
    if run and option == "Use Camera":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not access webcam.")
            st.stop()

    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        customer_id, customer_name = identify_face(frame, known_face_encodings, known_customer_ids, known_customer_names)
        if customer_id is None and manual_customer_id:
            customer_id = manual_customer_id
            customer_name = "Manual Entry"
        elif customer_id is None:
            customer_id = "UNKNOWN_" + str(uuid.uuid4())[:8]
            customer_name = "Unknown"
        customer_id_display.write(f"Customer ID: {customer_id} ({customer_name if customer_name else 'Name TBD'})")

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1) / 255.0
            
            predictions = model.predict(face)
            detected_emotion = emotion_dict[np.argmax(predictions)]
            emotion_counts[detected_emotion] += 1
            
            store_emotion(customer_id, detected_emotion, 1, "Uploaded Image", session_id)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        st.image(frame, channels="BGR", use_container_width=True)
        
       
        fig, ax = plt.subplots()
        ax.bar(emotion_counts.keys(), emotion_counts.values(), color=['red', 'green', 'blue', 'yellow', 'gray', 'purple', 'orange'])
        ax.set_xlabel("Emotions")
        ax.set_ylabel("Count")
        ax.set_title("Emotion Detection Stats (Uploaded Image)")
        ax.set_xticklabels(emotion_counts.keys(), rotation=45)
        stats_chart.pyplot(fig)
    
    
    if cap and cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if 'customer_id' in st.session_state and 'customer_name' in st.session_state:
                customer_id, customer_name = st.session_state.customer_id, st.session_state.customer_name
            else:
                customer_id, customer_name = identify_face(frame, known_face_encodings, known_customer_ids, known_customer_names)
                if customer_id is None and manual_customer_id:
                    customer_id = manual_customer_id
                    customer_name = "Manual Entry"
                elif customer_id is None:
                    customer_id = "UNKNOWN_" + str(uuid.uuid4())[:8]
                    customer_name = "Unknown"
                st.session_state.customer_id = customer_id
                st.session_state.customer_name = customer_name
            
            customer_id_display.write(f"Customer ID: {customer_id} ({customer_name if customer_name else 'Name TBD'})")

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48))
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=-1) / 255.0  

                predictions = model.predict(face)
                detected_emotion = emotion_dict[np.argmax(predictions)]
                emotion_counts[detected_emotion] += 1

                store_emotion(customer_id, detected_emotion, 1, "Camera", session_id)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, detected_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            stframe.image(frame, channels="BGR", use_container_width=True)

            fig, ax = plt.subplots()
            ax.bar(emotion_counts.keys(), emotion_counts.values(), color=['red', 'green', 'blue', 'yellow', 'gray', 'purple', 'orange'])
            ax.set_xlabel("Emotions")
            ax.set_ylabel("Count")
            ax.set_title("Live Emotion Detection Stats")
            ax.set_xticklabels(emotion_counts.keys(), rotation=45)
            stats_chart.pyplot(fig)
    
    if stop and cap:
        cap.release()

elif analysis_type == "Past Analysis":
    st.subheader("Past Data Analysis")
    
    
    known_face_encodings, known_customer_ids, known_customer_names = load_known_faces()
    
    conn = sqlite3.connect("emotion_data.db")
    c = conn.cursor()
    c.execute("SELECT DISTINCT customer_id FROM EmotionData")
    customer_ids = [row[0] for row in c.fetchall()]
    conn.close()

    customer_name_map = {cid: cname for cid, cname in zip(known_customer_ids, known_customer_names)}
    customer_options = ["All"] + [f"{cid} ({customer_name_map.get(cid, 'Unnamed')})" for cid in customer_ids]
    
    selected_customer_option = st.selectbox("Select Customer (or leave as 'All' for overall analysis)", customer_options)
    selected_customer_id = selected_customer_option.split(" (")[0] if selected_customer_option != "All" else "All"
    
    if selected_customer_id == "All":
        df = load_past_data()
    else:
        df = load_past_data(selected_customer_id)
    
    if df.empty:
        st.write("No data available for the selected customer.")
    else:
        if st.checkbox("Show Raw Data"):
            st.dataframe(df)

        emotion_counts_past = df.groupby("emotion")["count"].sum()

        st.subheader("Emotion Distribution (Pie Chart)")
        fig, ax = plt.subplots()
        ax.pie(emotion_counts_past, labels=emotion_counts_past.index, autopct="%1.1f%%", colors=['red', 'green', 'blue', 'yellow', 'gray', 'purple', 'orange'])
        ax.set_title(f"Emotion Distribution for {'All Customers' if selected_customer_id == 'All' else customer_name_map.get(selected_customer_id, 'Unnamed')}")
        st.pyplot(fig)

        st.subheader("Emotion Count (Bar Chart)")
        fig, ax = plt.subplots()
        ax.bar(emotion_counts_past.index, emotion_counts_past.values, color=['red', 'green', 'blue', 'yellow', 'gray', 'purple', 'orange'])
        ax.set_xlabel("Emotions")
        ax.set_ylabel("Count")
        ax.set_title(f"Emotion Count for {'All Customers' if selected_customer_id == 'All' else customer_name_map.get(selected_customer_id, 'Unnamed')}")
        ax.set_xticklabels(emotion_counts_past.index, rotation=45)
        st.pyplot(fig)

        
        st.subheader("Additional Insights")
        st.write(f"Total detections: {df['count'].sum()}")
        st.write(f"Most frequent emotion: {emotion_counts_past.idxmax()} ({emotion_counts_past.max()} times)")
        if selected_customer_id != "All":
            st.write(f"Data collected from {df['source'].nunique()} unique sources.")