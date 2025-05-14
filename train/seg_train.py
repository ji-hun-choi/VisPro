import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import os
import subprocess
from models.unet import UNet


def run(config):
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["learning_rate"]
    num_classes = config["num_classes"]
    input_size = config["input_size"]  # (H, W)
    save_path = config["output_path"]
    trt_engine_path = save_path.replace(".onnx", ".trt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor()
    ])

    # Dummy dataset (replace with real segmentation dataset if available)
    train_set = datasets.FakeData(
        size=100,
        image_size=(3, *input_size),
        num_classes=num_classes,
        transform=transform,
        target_transform=lambda t: torch.randint(0, num_classes, input_size)
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device).long()
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    model.eval()
    dummy_input = torch.randn(1, 3, *input_size).to(device)
    torch.onnx.export(model, dummy_input, save_path,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                      opset_version=11)
    print(f"ONNX 모델 저장 완료: {save_path}")

    print("TensorRT 엔진 변환 시작...")
    trtexec_cmd = [
        "trtexec",
        f"--onnx={save_path}",
        f"--saveEngine={trt_engine_path}",
        "--explicitBatch"
    ]
    try:
        subprocess.run(trtexec_cmd, check=True)
        print(f"TensorRT 엔진 저장 완료: {trt_engine_path}")
    except FileNotFoundError:
        print("trtexec 명령어를 찾을 수 없습니다.")
    except subprocess.CalledProcessError as e:
        print(f"TensorRT 변환 실패: {e}")
