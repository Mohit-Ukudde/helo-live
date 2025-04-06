import cv2
import numpy as np
import os
import torch
import time
from pathlib import Path
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image
import io
import csv
from datetime import datetime

class LicensePlateWebcamDetector:
    def __init__(self, model_path, device='0', conf_threshold=0.5, gemini_api_key=None):
        """
        Initialize the License Plate Detector for webcam usage
        
        Args:
            model_path: Path to the trained YOLOv8 model
            device: Device to run inference on ('0' for GPU, 'cpu' for CPU)
            conf_threshold: Confidence threshold for detections
            gemini_api_key: API key for Google's Gemini AI model for OCR
        """
        # Check if CUDA is available when device is set to GPU
        if device != 'cpu' and not torch.cuda.is_available():
            print("CUDA is not available, defaulting to CPU")
            device = 'cpu'
            
        self.device = device
        self.conf_threshold = conf_threshold
        
        # Load the YOLO model
        print(f"Loading model from {model_path} on device {device}")
        self.model = YOLO(model_path)
        
        # Class names
        self.class_names = ['ordinary', 'hsrp']
        self.class_colors = [(0, 255, 0), (0, 0, 255)]  # Green for ordinary, Blue for HSRP
        
        # For storing detection history
        self.detection_history = []
        self.max_history = 100  # Maximum number of detections to store
        
        # Initialize Gemini for OCR if API key is provided
        self.gemini_available = False
        if gemini_api_key:
            try:
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                self.gemini_available = True
                print("Gemini OCR initialized successfully")
            except Exception as e:
                print(f"Error initializing Gemini OCR: {e}")
    
    def detect(self, image):
        """
        Detect license plates in an image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            results: Model detection results
            annotated_img: Image with detection annotations
        """
        # Perform detection
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )
        
        # Get annotated image
        annotated_img = results[0].plot()
        
        return results[0], annotated_img
    
    def extract_license_plates(self, image, results):
        """
        Extract license plate regions from detection results
        
        Args:
            image: Original image
            results: Detection results from YOLO model
            
        Returns:
            plate_images: List of cropped license plate images
            plate_info: List of dictionaries with plate information
        """
        plate_images = []
        plate_info = []
        
        height, width = image.shape[:2]
        
        if len(results.boxes) == 0:
            return [], []
            
        for i, box in enumerate(results.boxes):
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            # Get class and confidence
            cls_id = int(box.cls)
            conf = float(box.conf)
            class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class {cls_id}"
            
            # Crop the license plate
            plate_img = image[y1:y2, x1:x2].copy()
            
            # Add to lists
            plate_images.append(plate_img)
            plate_info.append({
                'type': class_name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2),
                'plate_text': None  # Will be filled by OCR if available
            })
            
        return plate_images, plate_info

    def extract_text_with_gemini(self, plate_image):
        """
        Extract text from license plate image using Gemini Vision API
        
        Args:
            plate_image: Cropped license plate image (numpy array)
            
        Returns:
            plate_text: Extracted text from license plate
        """
        if not self.gemini_available:
            return None
            
        try:
            # Convert OpenCV image (numpy array) to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))
            
            # Create byte stream for image
            byte_stream = io.BytesIO()
            pil_image.save(byte_stream, format='JPEG')
            byte_stream.seek(0)
            
            # Prepare prompt for Gemini
            prompt = """
            Extract the license plate number from this image. 
            Return ONLY the text content of the license plate without any additional text or explanations.
            If no text is clearly visible, respond with 'UNREADABLE'.
            """
            
            # Generate response from Gemini Vision
            response = self.gemini_model.generate_content([prompt, {"mime_type": "image/jpeg", "data": byte_stream.getvalue()}])
            
            # Get plate text
            plate_text = response.text.strip()
            return plate_text
            
        except Exception as e:
            print(f"Error in Gemini OCR: {e}")
            return None
    
    def process_frame(self, frame, use_ocr=True, save_plates=False, output_dir="detected_plates"):
        """
        Process a single frame from webcam to detect license plates
        
        Args:
            frame: Input frame (numpy array)
            use_ocr: Whether to use OCR to extract text
            save_plates: Whether to save detected plates
            output_dir: Directory to save detected plates
            
        Returns:
            annotated_frame: Frame with annotations
            plate_info: Information about detected plates
        """
        # Make a copy of the frame
        frame_copy = frame.copy()
        
        # Detect license plates
        results, annotated_frame = self.detect(frame_copy)
        
        # Extract license plate regions
        plate_images, plate_info = self.extract_license_plates(frame_copy, results)
        
        # Extract text using Gemini OCR if available and enabled
        if use_ocr and self.gemini_available and plate_images:
            for i, plate_img in enumerate(plate_images):
                if plate_img.size > 0:
                    plate_text = self.extract_text_with_gemini(plate_img)
                    plate_info[i]['plate_text'] = plate_text
                    
                    # Add text to the annotated image
                    if plate_text:
                        x1, y1, x2, y2 = plate_info[i]['bbox']
                        cv2.putText(
                            annotated_frame,
                            f"Text: {plate_text}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2
                        )
        
        # Save detected plates if requested
        if save_plates and plate_images:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Get current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save each plate
            for i, (plate_img, info) in enumerate(zip(plate_images, plate_info)):
                if plate_img.size > 0:
                    plate_type = info['type']
                    conf = info['confidence']
                    plate_text = info.get('plate_text', 'unknown')
                    
                    # Create filename
                    filename = f"{timestamp}_plate{i+1}_{plate_type}_{plate_text}_{conf:.2f}.jpg"
                    save_path = os.path.join(output_dir, filename)
                    
                    # Save plate image
                    cv2.imwrite(save_path, plate_img)
                    
                    # Add to detection history with timestamp
                    detection_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'plate_type': plate_type,
                        'confidence': conf,
                        'plate_text': plate_text,
                        'image_path': save_path
                    }
                    self.detection_history.append(detection_entry)
                    
                    # Limit history size
                    if len(self.detection_history) > self.max_history:
                        self.detection_history.pop(0)
                    
                    # Save to CSV
                    self.save_detection_history(os.path.join(output_dir, "detection_history.csv"))
        
        # Add additional information to the frame
        self.add_info_to_frame(annotated_frame, plate_info)
        
        return annotated_frame, plate_info
    
    def add_info_to_frame(self, frame, plate_info):
        """Add information overlay to the frame"""
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            frame,
            f"Time: {timestamp}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Add number of detections
        cv2.putText(
            frame,
            f"Detections: {len(plate_info)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
    
    def save_detection_history(self, csv_path):
        """Save detection history to CSV file"""
        is_new_file = not os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            fieldnames = ['timestamp', 'plate_type', 'confidence', 'plate_text', 'image_path']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if is_new_file:
                writer.writeheader()
            
            # Write only the newest entry
            if self.detection_history:
                writer.writerow(self.detection_history[-1])
    
    def get_detection_history(self):
        """Get the detection history"""
        return self.detection_history

def init_webcam(camera_id=0, width=1280, height=720):
    """Initialize webcam with specified resolution"""
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap 