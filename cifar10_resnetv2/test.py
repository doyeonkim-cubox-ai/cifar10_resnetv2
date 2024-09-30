import torch
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import lightning as L
from cifar10_resnetv2.modlit import CIFARResNetV2
from cifar10_resnetv2.data import CIFAR10DataModule
import argparse


def main():
    # Add parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, help='Pick Model(ex.resnet110 / v2resnet110)')
    m = parser.parse_args().model

    dm = CIFAR10DataModule(data_dir="./cifar10", batch_size=128)
    dm.prepare_data()
    dm.setup(stage="test")

    checkpoint = f"./model/model_{m}.ckpt"
    net = CIFARResNetV2.load_from_checkpoint(checkpoint, m=m)
    trainer = L.Trainer(accelerator='cuda', devices=1)
    trainer.test(net, dm)


if __name__ == "__main__":
    main()
