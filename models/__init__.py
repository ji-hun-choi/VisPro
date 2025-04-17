from models.resnet import get_resnet
from models.efficientnet import get_efficientnet

MODEL_REGISTRY = {
    "resnet18": get_resnet,
    "efficientnet_b0": get_efficientnet,
}


def get_model(name, num_classes):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name](num_classes)