import train
import test
from torchvision.datasets import MNIST
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor
from mnist_net import MNISTNet
import torch.nn as nn
import matplotlib.pyplot as plt
import torch

def vis_tensor(dataset):
    tensor,_ = dataset[10]
    tensor = tensor.data.clone()
    tensor = tensor.squeeze()
    np_array = tensor.numpy()
    plt.imshow(np_array)
    plt.show()
    print(len(dataset))

train_mnist_data = MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True
)

test_mnist_data = MNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True
)

train_emnist_data = EMNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
    split="balanced"
)

test_emnist_data = EMNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True,
    split="balanced"
)

vis_tensor(train_mnist_data)
vis_tensor(test_mnist_data)
vis_tensor(train_emnist_data)
vis_tensor(test_emnist_data)


