import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import subprocess

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


def run(config):
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['learning_rate']
    save_path = config['output_path']
    trt_engine_path = save_path.replace(".onnx", ".trt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    model = SimpleCNN().to(device)
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
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 1, 28, 28).to(device)

        torch.onnx.export(
            model, dummy_input, "./models/cnn_mnist.onnx",
            export_params=True,
            opset_version=11,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            do_constant_folding=True
        )

        print(f"ONNX 모델 저장 완료: {save_path}")

    # TensorRT 변환 (trtexec 사용)
    print("TensorRT 엔진 변환 시작...")


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
        print("trtexec 명령어를 찾을 수 없습니다. TensorRT가 올바르게 설치되어 있는지 확인하세요.")
    except subprocess.CalledProcessError as e:
        print(f"TensorRT 변환 실패: {e}")