from ultralytics import YOLO

model = YOLO("/Users/virocchan/Downloads/AIshowcase/pill.pt")
model.export(format="onnx", imgsz=320)