from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='YOUR_FILE_ROUTING',
    epochs=30,
    imgsz=640,
    batch=16,
    device='cpu'
)