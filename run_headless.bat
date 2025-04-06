@echo off
echo Starting License Plate Webcam Detector (Headless Mode)
echo ===================================================
python webcam_detector.py --save --gemini-key YOUR_GEMINI_API_KEY_HERE --max-frames 50
echo.
echo Application completed. Results are saved in the detected_plates directory.
echo To see the results, check the detected_plates/frames folder for frame images
echo and the detected_plates folder for detected license plates.
pause 