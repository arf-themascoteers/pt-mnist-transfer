import train
import test
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor
from mnist_net import MNISTNet

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

model = train.train(model, train_emnist_data)
test.test(model, test_emnist_data)