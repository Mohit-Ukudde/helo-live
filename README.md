# License Plate Detection with Webcam

This application uses computer vision and OCR to detect and recognize license plates in real-time using your webcam.

## Features

- Real-time license plate detection using YOLOv8
- OCR text extraction from license plates using Google's Gemini AI
- Two modes: command-line webcam application and Streamlit web interface
- Detection history and statistics
- Save detected license plates for later analysis

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Make sure you have a trained YOLOv8 model (best.pt) in this directory or in the parent directory.

3. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Usage

### Command-line Webcam Application

1. Edit the `run_webcam.bat` file to include your Gemini API key.
2. Run the batch file:
   ```
   run_webcam.bat
   ```
3. Controls while running:
   - Press 'q' to quit
   - Press 's' to save the current frame
   - Press 'p' to toggle automatic plate saving

### Streamlit Web Application

1. Run the web application:
   ```
   run_streamlit.bat
   ```
   or manually:
   ```
   streamlit run app.py
   ```

2. Open your browser at http://localhost:8501
3. Configure settings in the sidebar
4. Click "Start Detection" to begin webcam feed
5. Detected license plates will be displayed and saved (if enabled)

## Output

All detected plates and history are saved in the `detected_plates` directory:
- License plate images with timestamps and detection information
- A CSV file with all detection records

## Command Line Options

The webcam detector supports several command-line options:

```
python webcam_detector.py --help
```

Options include:
- `--model`: Path to YOLOv8 model
- `--camera`: Camera index (default: 0)
- `--width`: Camera width (default: 1280)
- `--height`: Camera height (default: 720)
- `--conf`: Confidence threshold (default: 0.5)
- `--device`: Device to use ('0' for GPU, 'cpu' for CPU)
- `--save`: Save detected plates
- `--gemini-key`: Gemini API key for OCR
- `--output-dir`: Output directory for detections 