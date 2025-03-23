import cv2
import mediapipe as mp
import os
import numpy as np
from pygame import mixer
import time
from tensorflow.keras.models import load_model
import argparse
import json

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Initialize audio alert system
def setup_alert_system(sound_file='alert_tone.mp3'):
    mixer.init()
    try:
        mixer.music.load(sound_file)
        return True
    except:
        print(f"Warning: Could not load sound file {sound_file}")
        return False

# Load detection models and classifiers
def load_detection_models(eye_model_path, class_mapping_path=None):
    facial_state_model = load_model(eye_model_path)
    if class_mapping_path and os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
    else:
        class_mapping = {0: "closed", 1: "open", 2: "yawn", 3: "no_yawn"}
    return facial_state_model, class_mapping

# Process facial region for prediction
def process_facial_region(frame, x, y, w, h, target_size=(24, 24)):
    region = frame[y:y+h, x:x+w]
    if region.size == 0:
        return None
    region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    region_resized = cv2.resize(region_gray, target_size)
    region_normalized = region_resized / 255.0
    region_prepared = np.expand_dims(region_normalized.reshape(target_size[0], target_size[1], 1), axis=0)
    return region_prepared

# Detect and extract facial regions of interest
def detect_facial_regions(frame):
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * width), int(bboxC.ymin * height), \
                         int(bboxC.width * width), int(bboxC.height * height)
            eyes_h = int(h * 0.4)
            return {
                'face': (x, y, w, h),
                'left_eye': (x, y, w//2, eyes_h),
                'right_eye': (x + w//2, y, w//2, eyes_h),
                'mouth': (x, y + eyes_h, w, h - eyes_h)
            }
    return None

# Predict facial state
def predict_facial_states(frame, regions, model, class_mapping):
    if regions is None:
        return {'left_eye': 'unknown', 'right_eye': 'unknown', 'mouth': 'unknown'}
    results = {}
    for key in ['left_eye', 'right_eye', 'mouth']:
        region = regions[key]
        data = process_facial_region(frame, *region)
        if data is not None:
            pred = model.predict(data)
            results[key] = class_mapping.get(np.argmax(pred), 'unknown')
        else:
            results[key] = 'unknown'
    return results

# Main monitoring function
def monitor_alertness(facial_state_model, class_mapping, alert_sound=False, alert_threshold=15, cooldown_period=5):
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Could not open video source")
        return
    alertness_score = 0
    last_alert_time = 0
    prev_frame_time = 0
    new_frame_time = 0
    eye_closed_start_time = None  # Track when eyes first closed
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = new_frame_time
        regions = detect_facial_regions(frame)
        
        if regions is None:
            alertness_score = 0  # Reset score if no face detected
            eye_closed_start_time = None  # Reset eye closure tracking
            continue  # Skip processing if no face detected
        
        facial_states = predict_facial_states(frame, regions, facial_state_model, class_mapping)
        
        if facial_states.get('left_eye') == 'closed' and facial_states.get('right_eye') == 'closed':
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()  # Start tracking closure time
            elif time.time() - eye_closed_start_time >= 2:  # Check if eyes closed for 2+ sec
                if alert_sound and not mixer.music.get_busy():  # Play only if not already playing
                    mixer.music.play()
                eye_closed_start_time = None  # Reset timer after playing alert
        else:
            eye_closed_start_time = None  # Reset if eyes open
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Driver Alertness Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facial State Monitoring System')
    parser.add_argument('--model', type=str, default='facial_state_model.h5', help='Path to trained facial state model')
    parser.add_argument('--mapping', type=str, default='class_mapping.json', help='Path to class mapping JSON file')
    parser.add_argument('--alert-sound', type=str, default='alert_tone.mp3', help='Path to alert sound file')
    parser.add_argument('--threshold', type=int, default=15, help='Alertness threshold for triggering warnings')
    args = parser.parse_args()
    facial_state_model, class_mapping = load_detection_models(args.model, args.mapping)
    alert_sound = setup_alert_system(args.alert_sound)
    monitor_alertness(facial_state_model, class_mapping, alert_sound, alert_threshold=args.threshold)
