from ultralytics import YOLO

# Load YOLOv10s PyTorch model
model = YOLO('weights\yolov10s.pt')

# Export to ONNX format
model.export(format='onnx')
