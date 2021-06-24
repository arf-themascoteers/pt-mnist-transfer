import train
import test
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from mnist_net import MNISTNet

train_mnist_data = MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
)

test_mnist_data = MNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True,
)

model = MNISTNet()

model = train.train(model, train_mnist_data)
test.test(model, train_mnist_data)