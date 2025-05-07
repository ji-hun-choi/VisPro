import os
import subprocess
from ultralytics import YOLO


def run(config):
    model_name = config["model"]  # ex: yolov8n, yolov5n
    data_yaml = config["data_yaml"]  # path to dataset yaml, ex: ./data/yolo_coco128.yaml
    epochs = config["epochs"]
    batch = config["batch"]
    img_size = config["img_size"]
    onnx_path = config["output_path"]
    trt_path = onnx_path.replace(".onnx", ".trt")

    # Step 1: Load or define model
    if model_name.endswith(".pt"):
        model = YOLO(model_name)  # pretrained weights
    elif "yolov5" in model_name or "yolov11" in model_name:
        model = YOLO(model_name)  # predefined model name (e.g., yolov5n, yolov8n)
    else:
        model = YOLO(f"{model_name}.yaml")  # custom YAML

    # Step 2: Train
    model.train(data=data_yaml, epochs=epochs, imgsz=img_size, batch=batch)

    # Step 3 and 4: Export to ONNX and TensorRT
    result = model.export(format="engine", data='coco128.yaml', imgsz=img_size,
                          dynamic=True, simplify=True, int8=True, nms=True, batch=4)
    print(f"ONNX export 완료: {result}")

    try:
        print(f"TensorRT 엔진 저장 완료: {trt_path}")
    except FileNotFoundError:
        print("TensorRT(trtexec) 명령어를 찾을 수 없습니다.")
    except subprocess.CalledProcessError as e:
        print(f"TensorRT 변환 실패: {e}")
