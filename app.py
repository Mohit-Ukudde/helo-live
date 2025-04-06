import streamlit as st
import cv2
import os
import time
import numpy as np
import sys
import pandas as pd
from pathlib import Path
from PIL import Image
import threading
import queue
import shutil
import base64
from datetime import datetime
import tempfile

# Add parent directory to path to import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
sys.path.append(current_dir)  # Make sure utils can be imported from current dir

from utils import LicensePlateWebcamDetector, init_webcam

# Global variables
frame_queue = queue.Queue(maxsize=2)  # Queue to hold frames
stop_event = threading.Event()  # Event to signal thread to stop
processed_frame = None  # Most recent processed frame
detection_data = []  # Store detection data
output_dir = os.path.join(current_dir, "detected_plates")  # Directory to save detections

def webcam_thread(camera_id, width, height):
    """Thread to continuously read frames from webcam"""
    # Initialize webcam
    cap = init_webcam(camera_id, width, height)
    
    if not cap.isOpened():
        st.error(f"Error: Could not open camera {camera_id}")
        return
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
            
        # If queue is full, remove oldest frame
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
                
        # Add new frame to queue
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass
            
    # Release resources
    cap.release()

def get_image_base64(image_path):
    """Convert image to base64 string for HTML display"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def main():
    st.set_page_config(
        page_title="License Plate Detection",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("License Plate Detection System")
    st.sidebar.title("Settings")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector settings in session state if not exist
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'detection_running' not in st.session_state:
        st.session_state.detection_running = False
    if 'save_plates' not in st.session_state:
        st.session_state.save_plates = True
    if 'detections' not in st.session_state:
        st.session_state.detections = []
        
    # Sidebar settings
    model_path = None
    
    # First, check in current directory
    local_model = os.path.join(current_dir, "best.pt")
    parent_model = os.path.join(parent_dir, "best.pt")
    
    if os.path.exists(local_model):
        model_path = local_model
    elif os.path.exists(parent_model):
        # Copy the model to the local directory
        shutil.copy(parent_model, local_model)
        model_path = local_model
        
    if model_path:
        st.sidebar.success(f"Using model: {os.path.basename(model_path)}")
    else:
        st.sidebar.error("No model found. Please upload a model file.")
        uploaded_model = st.sidebar.file_uploader("Upload YOLO model (.pt)", type=['pt'])
        if uploaded_model:
            with open(local_model, "wb") as f:
                f.write(uploaded_model.getbuffer())
            model_path = local_model
            st.sidebar.success(f"Model uploaded successfully!")
    
    # Camera settings
    camera_id = st.sidebar.number_input("Camera ID", min_value=0, max_value=10, value=0)
    cam_width = st.sidebar.number_input("Camera Width", min_value=320, max_value=4096, value=1280)
    cam_height = st.sidebar.number_input("Camera Height", min_value=240, max_value=2160, value=720)
    
    # Detection settings
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    
    # Gemini API key for OCR
    gemini_key = st.sidebar.text_input("Gemini API Key (for OCR)", type="password")
    
    # Device selection
    device_options = ["CPU", "GPU"]
    device_selection = st.sidebar.selectbox("Device", device_options, index=1)
    device = "cpu" if device_selection == "CPU" else "0"
    
    # Initialize detector if model is available
    if model_path and st.session_state.detector is None:
        with st.spinner("Initializing detector..."):
            try:
                st.session_state.detector = LicensePlateWebcamDetector(
                    model_path=model_path,
                    device=device,
                    conf_threshold=conf_threshold,
                    gemini_api_key=gemini_key if gemini_key else None
                )
                st.success("Detector initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing detector: {e}")
    
    # Create main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Webcam feed display
        st.subheader("Live Detection")
        video_placeholder = st.empty()
        status_text = st.empty()
        
        # Start/Stop detection buttons
        detection_col1, detection_col2 = st.columns(2)
        with detection_col1:
            if not st.session_state.detection_running:
                if st.button("Start Detection", use_container_width=True):
                    # Start webcam thread
                    if st.session_state.detector is not None:
                        stop_event.clear()
                        thread = threading.Thread(
                            target=webcam_thread,
                            args=(camera_id, cam_width, cam_height),
                            daemon=True
                        )
                        thread.start()
                        st.session_state.detection_running = True
            else:
                if st.button("Stop Detection", use_container_width=True):
                    # Stop webcam thread
                    stop_event.set()
                    st.session_state.detection_running = False
        
        with detection_col2:
            # Toggle plate saving
            save_plates = st.checkbox("Save Detected Plates", value=st.session_state.save_plates)
            st.session_state.save_plates = save_plates
    
    with col2:
        # Detection history
        st.subheader("Detection History")
        history_placeholder = st.empty()
    
    # Process frames if detection is running
    if st.session_state.detection_running and st.session_state.detector is not None:
        fps_list = []  # Store recent FPS values
        
        # Main detection loop
        while st.session_state.detection_running:
            try:
                # Get frame from queue
                frame = frame_queue.get(timeout=1)
                
                # Process frame
                start_time = time.time()
                processed_frame, plate_info = st.session_state.detector.process_frame(
                    frame,
                    use_ocr=True if gemini_key else False,
                    save_plates=st.session_state.save_plates,
                    output_dir=output_dir
                )
                
                # Calculate FPS
                process_time = time.time() - start_time
                fps = 1.0 / process_time if process_time > 0 else 0
                fps_list.append(fps)
                if len(fps_list) > 10:
                    fps_list.pop(0)
                avg_fps = sum(fps_list) / len(fps_list)
                
                # Add FPS to the frame
                cv2.putText(
                    processed_frame,
                    f"FPS: {avg_fps:.1f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Convert to RGB for display
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Display processed frame
                video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                
                # Update status
                num_plates = len(plate_info)
                status = f"Detected {num_plates} license plate{'s' if num_plates != 1 else ''}"
                if num_plates > 0:
                    status += ": " + ", ".join([
                        f"{info['type']} ({info.get('plate_text', 'N/A')})" 
                        for info in plate_info
                    ])
                status_text.info(status)
                
                # Get detection history and display
                detection_history = st.session_state.detector.get_detection_history()
                
                if detection_history:
                    # Create DataFrame for display
                    df = pd.DataFrame(detection_history[-10:])  # Show last 10 detections
                    
                    # Sort by timestamp in descending order
                    if 'timestamp' in df.columns:
                        df = df.sort_values('timestamp', ascending=False)
                    
                    # Update history display
                    history_placeholder.dataframe(
                        df[['timestamp', 'plate_type', 'plate_text', 'confidence']],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Store in session state
                    st.session_state.detections = detection_history
                
            except queue.Empty:
                # No frame available, just continue
                video_placeholder.warning("Waiting for webcam...")
                time.sleep(0.1)
            except Exception as e:
                st.error(f"Error processing frame: {e}")
                time.sleep(0.5)
    
    # Display saved detections
    if not st.session_state.detection_running:
        st.subheader("Saved Detections")
        
        # Check for detected_plates directory and files
        if os.path.exists(output_dir):
            plate_images = list(Path(output_dir).glob("*.jpg"))
            plate_images = [p for p in plate_images if "frame_" not in p.name]  # Exclude full frames
            
            if plate_images:
                plate_images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                
                # Display in grid
                cols = st.columns(3)
                for i, img_path in enumerate(plate_images[:9]):  # Display up to 9 images
                    with cols[i % 3]:
                        st.image(str(img_path), caption=img_path.name, use_column_width=True)
            else:
                st.info("No saved detections found.")
        else:
            st.info("No detection directory found.")
        
        # Display detection history from CSV if available
        csv_path = os.path.join(output_dir, "detection_history.csv")
        if os.path.exists(csv_path):
            st.subheader("Detection Records")
            try:
                df = pd.read_csv(csv_path)
                st.dataframe(df, use_container_width=True)
                
                # Download button for CSV
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download Detection Data",
                    data=csv_data,
                    file_name="license_plate_detections.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error loading detection history: {e}")

if __name__ == "__main__":
    main() 