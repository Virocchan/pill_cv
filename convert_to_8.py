
import sys

print("PYTHON:", sys.executable)
print("VERSION:", sys.version)

from ultralytics import YOLO

model = YOLO("/Users/virocchan/Downloads/AIshowcase/pill.pt")

model.export(
    format="tflite",
    int8=True,
    imgsz=320
)