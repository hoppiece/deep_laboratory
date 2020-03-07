import torchvision
from torchvision import transforms

from laboratory.conf import Config

class MnistLoaderBase(object):
    pass

class CifarLoaderBase(object):
    def __init__(self):
        print("Loading data...")
        
        self._dset_type = {"cifar10"}

        self.transform = transforms.Compose(
            [transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )

        self.rawtrainset = torchvision.datasets.CIFAR10(
            root = Config.CIFAR10_PATH, 
            train = True,
            download = True
            )

        self.trainset = torchvision.datasets.CIFAR10(
            root = Config.CIFAR10_PATH, 
            train=True,
            download = True, 
            transform = self.transform
            )

        self.testset = torchvision.datasets.CIFAR10(
            root=Config.CIFAR10_PATH, 
            train=False,
            download=True, 
            transform = self.transform
            )

        self._load_train = None

        self._load_test = None

    @property
    def dset_type(self):
        return self._dset_type

    @property
    def load_train(self):
        return self._load_train

    @property
    def load_test(self):
        return self._load_test