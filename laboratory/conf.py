from pathlib import Path

BASE_PATH = Path(__file__).parent.resolve()

class Config(object):
    MNIST_PATH = BASE_PATH / "../datasets/mnist"
    CIFAR10_PATH = BASE_PATH / "../datasets/cifar10"