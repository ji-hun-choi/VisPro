from train import cnn_train, dnn_train, yolo_train, seg_train

TRAIN_REGISTRY = {
    "cnn": cnn_train.run,
    "dnn": dnn_train.run,
    "yolo": yolo_train.run,
    "seg": seg_train.run
}


def get_trainer(name):
    if name not in TRAIN_REGISTRY:
        raise ValueError(f"Unknown train: {name}")
    return TRAIN_REGISTRY[name]