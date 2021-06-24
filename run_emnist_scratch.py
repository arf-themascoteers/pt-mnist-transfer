import train
import test
from torchvision.datasets import EMNIST
from torchvision.transforms import ToTensor
from emnist_net import EMNISTNet

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

model = EMNISTNet()

model = train.train(model, train_emnist_data)
test.test(model, train_emnist_data)