from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='C:/Users/carlos.segundo/source/workspace/Python couser/Brazil-Plates-Detector-1',
    epochs=30,
    imgsz=640,
    batch=16,
    device='cpu'
)