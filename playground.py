from mnist_net import MNISTNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

NUM_EPOCHS = 3
BATCH_SIZE = 1000

working_set = datasets.EMNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
    split='balanced'
)

dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=True)
model = MNISTNet()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

for epoch in range(0, NUM_EPOCHS):
    for data, y_true in dataloader:
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.nll_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()
        print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')

