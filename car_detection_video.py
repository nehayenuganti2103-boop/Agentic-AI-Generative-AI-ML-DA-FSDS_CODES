import streamlit as st
import cv2
import tempfile
import numpy as np

st.set_page_config(page_title="Car Counter App", layout="wide")
st.title("🚗 Video Car Detection & Counter")

st.sidebar.header("Detection Settings")
scale_factor = st.sidebar.slider("Scale Factor", 1.1, 1.5, 1.1)
min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 3)

car_cascade = cv2.CascadeClassifier('C:/Users/Neha/OneDrive/Desktop/VSCode/OpenCV/Haarcascade/haarcascade_car.xml')

uploaded_file = st.file_uploader("Upload a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st_frame = st.empty()
    with col2:
        st.subheader("Current Statistics")
        count_display = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        cars = car_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
        
        car_count = len(cars)
        
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Car", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, use_container_width=True)
        count_display.metric("Cars in Frame", car_count)

    cap.release()
    st.success("Processing Finished!")
else:
    st.info("Upload a video to begin detection.")


