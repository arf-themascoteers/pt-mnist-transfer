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
    for i in range(5):
        tensor,_ = dataset[i]
        tensor = tensor.data.clone()
        tensor = tensor.squeeze()
        np_array = tensor.numpy()
        plt.imshow(np_array)
        plt.show()
    print(dataset)


train_mnist_data = EMNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
    split="mnist"
)

test_mnist_data = EMNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True,
    split="mnist"
)

train_digit_data = EMNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
    split="digits"
)

test_digit_data = EMNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True,
    split="digits"
)

vis_tensor(train_mnist_data)
vis_tensor(test_mnist_data)
vis_tensor(train_digit_data)
vis_tensor(test_digit_data)


