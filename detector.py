from ultralytics import YOLO
import cv2

model = YOLO("best.pt")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(
            frame,
            verbose=False
        )
        annotated_frame = results[0].plot()
        occupied = 0
        empty = 0
        boxes = results[0].boxes
        if boxes is not None:
            classes = boxes.cls.cpu().numpy().astype(int)
            for cls in classes:
                if cls == 0:
                    occupied += 1
                elif cls == 1:
                    empty += 1
        total = occupied + empty
        occupancy = (occupied / total * 100) if total > 0 else 0
        overlay = annotated_frame.copy()
        annotated_frame = cv2.addWeighted(
            overlay,
            0.4,
            annotated_frame,
            0.6,
            0
        )
        yield (
            annotated_frame,
            occupied,
            empty,
            total,
            occupancy
        )
    cap.release()

def process_image(image_path):
    image = cv2.imread(image_path)
    results = model.predict(
        image,
        verbose=False
    )
    annotated = results[0].plot()
    occupied = 0
    empty = 0
    boxes = results[0].boxes
    if boxes is not None:
        classes = boxes.cls.cpu().numpy().astype(int)
        for cls in classes:
            if cls == 0:
                occupied += 1
            elif cls == 1:
                empty += 1
    total = occupied + empty
    occupancy = (occupied / total * 100) if total > 0 else 0
    return (
        annotated,
        occupied,
        empty,
        total,
        occupancy
    )
