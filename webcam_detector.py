import cv2
import argparse
import os
import time
import sys
from pathlib import Path
import shutil

# Add parent directory to path so we can import from the utils module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils import LicensePlateWebcamDetector, init_webcam

def main():
    parser = argparse.ArgumentParser(description="License Plate Webcam Detector")
    parser.add_argument('--model', type=str, default=None, help='Path to YOLOv8 model')
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    parser.add_argument('--width', type=int, default=1280, help='Camera width')
    parser.add_argument('--height', type=int, default=720, help='Camera height')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='0', help='Device to use (0 for GPU, cpu for CPU)')
    parser.add_argument('--save', action='store_true', help='Save detected plates')
    parser.add_argument('--gemini-key', type=str, default=None, help='Gemini API key for OCR')
    parser.add_argument('--output-dir', type=str, default='detected_plates', help='Output directory for detections')
    parser.add_argument('--max-frames', type=int, default=100, help='Maximum number of frames to process')
    args = parser.parse_args()
    
    # Find model if not specified
    if args.model is None:
        # First check the local directory
        local_model = os.path.join(current_dir, "best.pt")
        parent_model = os.path.join(parent_dir, "best.pt")
        
        if os.path.exists(local_model):
            args.model = local_model
            print(f"Using local model: {local_model}")
        elif os.path.exists(parent_model):
            # Copy the model to the local directory
            shutil.copy(parent_model, local_model)
            args.model = local_model
            print(f"Copied model from parent directory: {local_model}")
        else:
            # Look for trained model in the standard location
            model_paths = list(Path(parent_dir).glob('runs/detect/*/weights/best.pt'))
            if not model_paths:
                print("No trained model found. Please specify a model path with --model")
                return
            args.model = str(model_paths[0])
            print(f"Using model: {args.model}")
    
    # Initialize detector
    detector = LicensePlateWebcamDetector(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        gemini_api_key=args.gemini_key
    )
    
    # Initialize webcam
    cap = init_webcam(args.camera, args.width, args.height)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return
    
    print("Starting webcam detection...")
    print(f"Saving frames and detections to {args.output_dir}")
    print(f"Will process up to {args.max_frames} frames")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Flag to control saving plates
    save_plates = args.save
    
    try:
        frame_count = 0
        while frame_count < args.max_frames:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error reading from webcam")
                break
            
            # Process frame
            start_time = time.time()
            processed_frame, plate_info = detector.process_frame(
                frame, 
                use_ocr=True, 
                save_plates=save_plates,
                output_dir=args.output_dir
            )
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            
            # Add FPS to the frame
            cv2.putText(
                processed_frame,
                f"FPS: {fps:.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Add save status to the frame
            save_status = "ON" if save_plates else "OFF"
            cv2.putText(
                processed_frame,
                f"Plate Saving: {save_status}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Save the frame instead of displaying it
            frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, processed_frame)
            
            # Print detection info
            num_plates = len(plate_info)
            print(f"Frame {frame_count}: Detected {num_plates} license plate(s), FPS: {fps:.2f}")
            
            for i, info in enumerate(plate_info):
                plate_text = info.get('plate_text', 'N/A')
                print(f"  Plate {i+1}: {info['type']} (conf: {info['confidence']:.2f}) - Text: {plate_text}")
            
            frame_count += 1
            
            # Small delay to simulate real-time processing
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Release resources
        cap.release()
        print(f"Webcam detection stopped after processing {frame_count} frames")
        print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 