これは別に意味ないので気にしないでください。。。

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import MultiStepLR

import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Model(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(Model, self).__init__()

        self.logreg = nn.Sequential(
            nn.Linear(input_size, output_size, dtype=torch.float),
        )

    def forward(self, x):
        return self.logreg(x)

class FtDataset(Dataset):
    def __init__(self, train_x, train_y) -> None:
        super(FtDataset, self).__init__()

        self.train_x = (train_x - train_x.mean()) / train_x.std()
        # self.train_x = train_x
        self.train_y = train_y
        self.len = train_x.shape[0]

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        x = train_x.loc[index,:]
        y = train_y.loc[index,:]

        return torch.Tensor(x), torch.Tensor(y)

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath')
    args = parser.parse_args()

    df = pd.DataFrame()
    try:
        df = pd.read_csv(args.filepath)
    except FileNotFoundError:
        print(f'No such file or directory: \'{args.filepath}\'', file=sys.stderr)
        exit(1)

    df = df.dropna().drop(columns=['Index']).reset_index(drop=True)
    train_x = df.select_dtypes(include=[np.number])

    ohe_hogwarts_house = OneHotEncoder()
    hogwarts_house = ohe_hogwarts_house.fit_transform(df[['Hogwarts House']]).toarray()
    train_y = pd.DataFrame(hogwarts_house, columns=ohe_hogwarts_house.categories_)

    epoch = 100
    train_dataset = FtDataset(train_x=train_x, train_y=train_y)
    test_dataset = FtDataset(train_x=train_x, train_y=train_y)
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0, drop_last=True)


    model = Model(input_size=train_x.shape[1], output_size=len(ohe_hogwarts_house.categories_[0]))
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1, verbose=True)
    # torch.autograd.set_detect_anomaly(True)

    for i in range(epoch):
        running_loss = 0.
        model.train()
        for batch, (x, y) in enumerate(train_dataloader):
            pred = model(x)
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        model.eval()
        with torch.no_grad():
            test_loss, correct = 0, 0
            size = len(test_dataloader.dataset)
            num_batches = len(test_dataloader)

            for x, y in test_dataloader:

                pred = model(x)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

        print(f"epoch {i}: loss {running_loss:>7f}")
