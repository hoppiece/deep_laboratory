import torch
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms

from laboratory.dloader.dloader_base import CifarLoaderBase
from laboratory.conf import Config


class CifarSamlpleLoader():
    def __init__(self):

        print("Loading data...")
        
        self._dset_type = {"cifar10"}

        self.transform = transforms.Compose(
            [transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )

        self.rawtrainset = datasets.CIFAR10(
            root = Config.CIFAR10_PATH, 
            train = True,
            download = True
            )

        self.trainset = datasets.CIFAR10(
            root = Config.CIFAR10_PATH, 
            train=True,
            download = True, 
            transform = self.transform
            )

        self.testset = datasets.CIFAR10(
            root=Config.CIFAR10_PATH, 
            train=False,
            download=True, 
            transform = self.transform
            )

        self.load_train = torch.utils.data.DataLoader(
            self.trainset,
            batch_size = 4,
            shuffle = True, 
            num_workers = 2
        )

        self.load_test = torch.utils.data.DataLoader(
            self.testset, 
            batch_size = 4,
            shuffle = False, 
            num_workers = 2
        )
