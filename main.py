import argparse
import yaml
import os
import urllib.request
from train import get_trainer


def load_config(model_name):
    config_path = os.path.join("config", f"{model_name}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_file(url, dest_path):
    try:
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")


def check_and_download_data(model_name):
    data_dir = os.path.join("data", model_name)
    os.makedirs(data_dir, exist_ok=True)

    urls = {
        "cnn_mnist": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        "dnn_fashion": "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz",
        "dnn_cifar10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    }

    if model_name in urls:
        filename = os.path.basename(urls[model_name])
        dest_file = os.path.join(data_dir, filename)

        if not os.path.exists(dest_file):
            download_file(urls[model_name], dest_file)
        else:
            print(f"Dataset already exists at {dest_file}")
    else:
        print(f"No automatic dataset available for {model_name}. Please place the dataset in {data_dir}")

def main():
    parser = argparse.ArgumentParser(description="VisPro Trainer")
    parser.add_argument('--algorithm', type=str, required=True,
                        help="algorithm name (e.g., cnn, dnn, yolo)")
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset name (e.g., mnist, fashion, cifar10)")
    parser.add_argument('--model', type=str, required=True,
                        help="Model name (e.g., cnn, resnet18, efficientnet_b0, yolov5)")
    args = parser.parse_args()

    dataset_name = args.algorithm + "_" + args.dataset

    config = load_config(dataset_name)
    config["model"] = args.model

    check_and_download_data(dataset_name)

    get_trainer(args.algorithm)(config)


if __name__ == "__main__":
    main()