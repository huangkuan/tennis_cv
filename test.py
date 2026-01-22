from ultralytics import YOLO
import cv2

input_video = "input_videos/test1.mov"
output_video = "output_videos/output_annotated.mp4"
output_csv = "output_videos/tracks.csv"

model = YOLO("yolo26n.pt")

# Set up video capture and writer
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {input_video}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
frame_idx = 0

# If you are only tracking the object on each frame separately, you can use model.predict instead of model.track
#for result in model.predict(source=input_video, stream=True, classes=[0, 32]):  # 0=person, 32=tennis ball in COCO
for result in model.track(source=input_video, stream=True, 
                          persist=True, classes=[0,32], conf=0.25, 
                          tracker="bytetrack.yaml", verbose=False):
    annotated_frame = result.plot()  # numpy array with boxes drawn (BGR)

    # Safety check (rare but good practice)
    if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
       annotated_frame = cv2.resize(annotated_frame, (width, height))

    writer.write(annotated_frame)

# --- Extract tracking info ---
    boxes = result.boxes
    if boxes is not None and boxes.id is not None:
        for box, tid, cls, conf in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy(),
            boxes.cls.cpu().numpy(),
            boxes.conf.cpu().numpy()
        ):
            x1, y1, x2, y2 = map(int, box)
            print({
                "frame": frame_idx,
                "track_id": int(tid),
                "class_id": int(cls),
                "class_name": model.names[int(cls)],
                "conf": float(conf),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })

    frame_idx += 1

