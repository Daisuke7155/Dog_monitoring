import cv2
import time

cap = cv2.VideoCapture(0)  # USBcamera

if not cap.isOpened():
    print("No camera")
    exit()

frame_count = 0
start_time = time.time()

while frame_count < 100:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

end_time = time.time()
elapsed_time = end_time - start_time
fps = frame_count / elapsed_time

print(f"average: {fps:.2f} FPS")

cap.release()
