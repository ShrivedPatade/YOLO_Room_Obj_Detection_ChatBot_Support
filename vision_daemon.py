# 


import time
import json
import os
import cv2
from datetime import datetime
from vision.scanner import scan_frame
from config import TRACKABLE_OBJECTS, MEMORY_FILE, SCAN_INTERVAL 

# Map your video files here
VIDEO_SOURCES = {
    "Living Room": "living_room_test.mpeg",
    "Kitchen": "kitchen_test.mpeg"
}

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

def run_video_daemon():
    print(f"--- Starting Video Testing Daemon (Simulating {SCAN_INTERVAL}s intervals) ---")
    
    # Open both videos
    caps = {room: cv2.VideoCapture(path) for room, path in VIDEO_SOURCES.items()}
    
    while True:
        try:
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            memory = load_memory()
            
            for room_name, cap in caps.items():
                if not cap.isOpened():
                    continue

                # Read the frame
                ret, frame = cap.read()
                
                # If video ends, loop back to the beginning
                if not ret:
                    print(f"[{room_name}] Video ended. Restarting loop...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    
                if ret:
                    # 1. Process the frame with YOLO
                    found_objects, annotated_frame = scan_frame(frame, TRACKABLE_OBJECTS)
                    
                    # 2. Save the image so you can visually verify what YOLO saw
                    cv2.imwrite(f"{room_name.replace(' ', '_')}_latest.jpg", annotated_frame)
                    
                    # 3. Update the memory dictionary
                    if room_name not in memory:
                        memory[room_name] = {"last_scanned": "", "objects": {}}
                        
                    memory[room_name]["last_scanned"] = now_str
                    
                    for obj in found_objects:
                        obj_name = obj['object']
                        memory[room_name]["objects"][obj_name] = {
                            "confidence": obj['confidence'],
                            "last_seen": now_str
                        }
                    
                    print(f"[{now_str}] {room_name}: Logged {len(found_objects)} objects.")
                    
                    # 4. Fast-forward the video to simulate time passing (e.g., skip 10 seconds)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps > 0:
                        frames_to_skip = int(fps * SCAN_INTERVAL)
                        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + frames_to_skip)

            # Save the updated JSON for the LLM to read
            save_memory(memory)
            
        except Exception as e:
            print(f"[Daemon Error] {e}")
            
        # Pause briefly before taking the next simulated snapshot
        time.sleep(2) 

if __name__ == "__main__":
    run_video_daemon()