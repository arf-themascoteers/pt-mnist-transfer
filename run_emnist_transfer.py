import train
import test
from torchvision.datasets import MNIST
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor
from mnist_net import MNISTNet
import torch.nn as nn

train_mnist_data = MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True
)

train_emnist_data = EMNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
    split="digits"
)

test_emnist_data = EMNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True,
    split="digits"
)

model = MNISTNet()
model = train.train(model, train_mnist_data)
model.linear1.requires_grad = False
model = train.train(model, train_emnist_data)
test.test(model, test_emnist_data)