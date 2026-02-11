import streamlit as st
import cv2
import tempfile
import numpy as np

st.set_page_config(page_title="Pedestrian Counter App", layout="wide")
st.title("🚶 Pedestrian Detection & Real-time Counter")

pedestrian_cascade = cv2.CascadeClassifier(r'C:/Users/Neha/OneDrive/Desktop/VSCode/OpenCV/Haarcascade/haarcascade_fullbody.xml')

st.sidebar.header("Model Parameters")
scale_factor = st.sidebar.slider("Scale Factor (Lower is more sensitive)", 1.01, 1.5, 1.1)
min_neighbors = st.sidebar.slider("Min Neighbors (Higher reduces false positives)", 1, 10, 4)

uploaded_file = st.file_uploader("Upload a video for pedestrian detection...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st_frame = st.empty()
    with col2:
        st.subheader("Live Stats")
        person_count_metric = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        pedestrians = pedestrian_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
        
        count = 0
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            count += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, use_container_width=True)
        person_count_metric.metric("Persons in Frame", count)

    cap.release()
    st.success("Analysis Complete!")
else:
    st.info("Please upload a video to start the detection process.")
