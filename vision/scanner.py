# import cv2
# import numpy as np
# import requests
# from ultralytics import YOLOWorld
# import json

# # Your verified USB tethering IP
# CAMERA_URL = "http://10.90.222.18:8080/shot.jpg"

# print("Loading YOLO-World (Large) in memory...")
# # Ensure you have yolov8l-world.pt downloaded in your root folder
# model = YOLOWorld('../yolov8l-world.pt') 

# def extract_visual_context(results):
#     """Parses YOLO output into a clean list of dictionaries."""
#     detected_objects = []
#     for box in results[0].boxes:
#         class_id = int(box.cls[0])
#         class_name = model.names[class_id]
#         confidence = float(box.conf[0])
#         coords = box.xyxy[0].tolist() 
        
#         detected_objects.append({
#             "object": class_name,
#             "confidence": round(confidence, 2),
#             "coordinates": [round(c, 2) for c in coords]
#         })
#     return detected_objects

# def scan_room(target_classes: list):
#     """
#     Dynamically sets what YOLO should look for, snaps a photo, and returns objects.
#     """
#     print(f"[Eyes] Looking for: {target_classes}")
#     model.set_classes(target_classes)
    
#     try:
#         # Request a single high-res snapshot
#         resp = requests.get(CAMERA_URL, timeout=5)
#         img_arr = np.array(bytearray(resp.content), dtype=np.uint8)
#         frame = cv2.imdecode(img_arr, -1)
        
#         # Run inference
#         results = model.predict(frame, conf=0.15) # Low confidence to catch more
        
#         # Optional: Save the annotated frame to disk so you can see what it saw
#         annotated_frame = results[0].plot()
#         cv2.imwrite("latest_memory.jpg", annotated_frame)
        
#         return extract_visual_context(results)
        
#     except Exception as e:
#         print(f"[Eyes] Error capturing image: {e}")
#         return []
    

import cv2
from ultralytics import YOLOWorld

print("Loading YOLO-World (Large) in memory...")
model = YOLOWorld('yolov8l-world.pt') 

# Keep track of the current classes to prevent PyTorch device crashes
_current_classes = []

def extract_visual_context(results):
    """Parses YOLO output into a clean list of dictionaries."""
    detected_objects = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        coords = box.xyxy[0].tolist() 
        
        detected_objects.append({
            "object": class_name,
            "confidence": round(confidence, 2),
            "coordinates": [round(c, 2) for c in coords]
        })
    return detected_objects

def scan_frame(frame, target_classes: list):
    """
    Takes a cv2 frame, runs YOLO-World, and returns the objects and annotated image.
    """
    global _current_classes
    
    # ONLY update the model classes if the list actually changed.
    # This prevents the CPU/CUDA tensor crash and speeds up the loop immensely.
    if target_classes != _current_classes:
        model.set_classes(target_classes)
        _current_classes = target_classes.copy()
    
    try:
        # Run inference
        results = model.predict(frame, conf=0.15, verbose=False) # Turned off verbose to keep terminal clean
        
        # Get the drawn image so the daemon can save it for debugging
        annotated_frame = results[0].plot()
        
        return extract_visual_context(results), annotated_frame
        
    except Exception as e:
        print(f"[Eyes Error] {e}")
        return [], frame