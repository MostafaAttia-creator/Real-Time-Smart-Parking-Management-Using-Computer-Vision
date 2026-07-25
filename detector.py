from ultralytics import YOLO
import cv2
import time

model = YOLO("best.pt")


def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    parking_start_time = {}

    if not cap.isOpened():
        return

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            verbose=False
        )

        annotated_frame = results[0].plot()

        occupied = 0
        empty = 0

        boxes = results[0].boxes

        if boxes is not None and boxes.id is not None:

            ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()

            current_time = time.time()

            for box, cls, track_id in zip(xyxy, classes, ids):

                x1, y1, x2, y2 = map(int, box)

                if cls == 0:

                    occupied += 1

                    if track_id not in parking_start_time:
                        parking_start_time[track_id] = current_time

                    elapsed = current_time - parking_start_time[track_id]

                    hours = int(elapsed // 3600)
                    minutes = int((elapsed % 3600) // 60)
                    seconds = int(elapsed % 60)

                    cv2.putText(
                        annotated_frame,
                        f"{hours:02}:{minutes:02}:{seconds:02}",
                        (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

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
