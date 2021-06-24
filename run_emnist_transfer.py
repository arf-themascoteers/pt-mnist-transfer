import train
import test
from torchvision.datasets import MNIST
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor
from mnist_net import MNISTNet

train_mnist_data = EMNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
    split="mnist"
)

train_digits_data = EMNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
    split="digits"
)

test_digits_data = EMNIST(
    root='data',
    train=False,
    transform=ToTensor(),
    download=True,
    split="digits"
)

model = MNISTNet()
model = train.train(model, train_mnist_data)
for param in model.linear1.parameters():
    param.requires_grad = False
model = train.train(model, train_digits_data)
test.test(model, test_digits_data)