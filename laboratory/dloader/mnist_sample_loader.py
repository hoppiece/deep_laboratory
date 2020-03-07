import torch 
import torch.utils.data
from torchvision import datasets
from torchvision.transforms import transforms

from laboratory.conf import Config


class MnistSampleLoader():
    def __init__(self):

        print("Loading data...")

        self._dset_type = {"mnist"}

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))]
            )

        self.rawtrainset = datasets.MNIST(
            root = Config.MNIST_PATH,
            train = True,
            download = True
        )

        self.trainset = datasets.MNIST(
            root = Config.MNIST_PATH,
            train = True,
            download = True,
            transform = self.transform
        )

        self.testset = datasets.MNIST(
            root = Config.MNIST_PATH,
            train = False, 
            download = True,
            transform = self.transform
        )

        self.load_train = torch.utils.data.DataLoader(
            self.trainset,
            batch_size = 100,
            shuffle = True,
            num_workers = 2
        )

        self.load_test = torch.utils.data.DataLoader(
            self.testset,
            batch_size = 100,
            shuffle = False,
            num_workers = 2
        )

