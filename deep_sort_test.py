import os
import random

import cv2
from ultralytics import YOLO

from deepsort_tracker import Tracker


video_path = '659e51d0cb3464.03930276.mkv'
video_out_path = '659e51d0cb3464.03930276.mkv'

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'H264'), cap.get(cv2.CAP_PROP_FPS),
                          (frame.shape[1], frame.shape[0]))

model = YOLO("yolov8n.pt")
model.fuse()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

unique_people_ids = set()

detection_threshold = 0.3
people_count = 0
print(frame_count)
total_frames = 0
while ret:
    if total_frames % 2 == 0:

        results = model(frame, verbose=False)

        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                if score > detection_threshold and class_id == 0:
                    detections.append([x1, y1, x2, y2, score])

            tracker.update(frame, detections)

            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = map(int, bbox)
                track_id = track.track_id

                if track_id not in unique_people_ids and class_id == 0:
                    unique_people_ids.add(track_id)

                    # Создаем новое изображение, объединяя полный кадр с выделенным участком
                    combined_image = frame.copy()
                    cv2.rectangle(combined_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Сохраняем изображение
                    cv2.imwrite(f"images/combined_frame_with_person_{track_id}.jpg", combined_image)


    cap_out.write(frame)
    ret, frame = cap.read()
    total_frames += 1
    if total_frames % 200 == 0:
        print(F"Всего уник. Клиентов: {unique_people_ids}")
        print(total_frames)

print(unique_people_ids)
cap.release()
cap_out.release()
cv2.destroyAllWindows()
print(unique_people_ids)