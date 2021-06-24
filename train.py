from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

def train(model, working_set):
    NUM_EPOCHS = 1
    BATCH_SIZE = 10
    dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss = None
    print("Training started...")
    for epoch  in range(0, NUM_EPOCHS):
        for data, y_true in dataloader:
            optimizer.zero_grad()
            y_pred = model(data)
            loss = F.nll_loss(y_pred, y_true)
            loss.backward()
            optimizer.step()
        print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    print("Training end")
    return model




