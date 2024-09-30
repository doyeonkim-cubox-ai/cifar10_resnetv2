import lightning as L
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
            ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
        ])

    def prepare_data(self):
        torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            self.train = torchvision.datasets.CIFAR10(self.data_dir, train=True, transform=self.transform_train)
            self.train, self.valid = random_split(self.train, [0.9, 0.1])
        if stage == "test":
            self.test = torchvision.datasets.CIFAR10(self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=8)
