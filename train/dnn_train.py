import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from models import get_model
import os
import subprocess

def run(config):
    model_name = config["model"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["learning_rate"]
    save_path = config["output_path"]
    trt_engine_path = save_path.replace(".onnx", ".trt")
    num_classes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("./data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    model = get_model(model_name, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{model_name}] Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model, dummy_input, save_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11, export_params=True
    )
    print(f"ONNX 모델 저장 완료: {save_path}")

    # TensorRT 변환
    trtexec_cmd = [
        "trtexec",
        f"--onnx={save_path}",
        f"--saveEngine={trt_engine_path}",
        f"--fp16"
    ]

    try:
        subprocess.run(trtexec_cmd, check=True)
        print(f"TensorRT 엔진 저장 완료: {trt_engine_path}")
    except FileNotFoundError:
        print("trtexec 명령어를 찾을 수 없습니다.")
    except subprocess.CalledProcessError as e:
        print(f"TensorRT 변환 실패: {e}")
