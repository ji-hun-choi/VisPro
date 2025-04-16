# 🚀 VisPro - Vision AI Multi-Model Project

**VisPro**는 다양한 비전 AI 모델(CNN, DNN, Object Detection, Segmentation)을  
Python으로 학습하고 C++에서 추론하는 구조로 설계된 멀티모델 프로젝트입니다.

> **Train in Python · Infer in C++**  
> One project, multiple model types — unified under VisPro.

---

## 🧠 프로젝트 개요

- ✅ 다양한 딥러닝 모델을 하나의 플랫폼에서 학습 및 추론
- ✅ Python 기반 PyTorch 학습 → ONNX 변환 → C++ 추론
- ✅ CNN, DNN, YOLO, Segmentation까지 확장 가능
- ✅ 추론 속도 및 효율성을 위한 C++ 백엔드 구성

---

## 🗂️ 프로젝트 구조 (초기 설계)

```
VisPro/
├── train/           # Python 학습 스크립트
├── models/          # Export된 ONNX 모델들
├── include/         # C++ 헤더 파일 (모델별 추론 클래스)
├── src/             # C++ 구현부
└── CMakeLists.txt   # 빌드 설정 파일
```

---

## 🔧 사용 환경

### Python (Training)
- Python 3.10+
- PyTorch
- ONNX
- OpenCV
- NumPy

### C++ (Inference)
- ONNX Runtime 또는 OpenCV DNN
- CMake 3.14+
- OpenCV 4.x 이상
- (옵션) TensorRT 지원

---

## 🧪 지원 모델 타입

- [x] CNN (분류기)
- [x] DNN (Fully Connected)
- [x] Object Detection (YOLOvX 등)
- [x] Segmentation (e.g. DeepLab, UNet)
- [ ] (추가 예정) Pose Estimation

---

## ⚙️ 빌드 및 실행 (C++ Inference)

```bash
mkdir build && cd build
cmake ..
make
```

실행 방법 및 예시는 추후 각 모델별 예제 코드에 추가 예정입니다.

---

## ✅ 개발 예정 항목 (TODO)

- [ ] 모델별 추론 클래스 분리 (예: `CNNInfer`, `Detector`)
- [ ] 공통 추론 인터페이스 설계
- [ ] 성능 측정 툴 및 시각화
- [ ] WebUI 또는 영상 출력 연동

---

## 👨‍💻 작성자

> 개발자: [Your Name]  
> 비전 AI 통합 프로젝트 VisPro는 실무 및 포트폴리오용으로 제작되었습니다.
