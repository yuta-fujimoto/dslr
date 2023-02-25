import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class LogisticRegression:
    def __init__(self, W = None) -> None:
        # weight and bias
        self.W = W
        self.mean = 0
        self.std = 0.

    def fit(self, x_df, y_df, learning_rate = 0.01, epoch = 10):
        x = x_df.values

        # normalize to optimize learning and prevent overflow 
        self.mean = x_df.mean().values
        self.std = x_df.std().values
        x = (x - self.mean) / self.std
        
        # concatenate bias
        x = np.concatenate([x, np.ones((x.shape[0], 1))], 1)
        y = y_df.values

        self.W = np.zeros((x.shape[1], y.shape[1]))

        size = x.shape[0]
        for i in range(epoch):
            activation = 1. / (1. + np.exp(-1. * np.matmul(self.W.T, x.T)))

            loss = -np.diag(np.matmul(y.T, np.log(activation.T)) + np.matmul(1 - y.T, np.log(1 - activation.T)))
            loss = loss / size

            grad = np.matmul((activation - y.T), x) / size

            self.W = self.W - grad.T * learning_rate

            correct = (activation.T.argmax(1) == y.argmax(1)).sum()
            print(f'epoch {i + 1}:\n loss: {loss} acc: {correct / size:.5}')

    def predict(self, x_df):
        x = x_df.values

        # concatenate bias
        x = np.concatenate([x, np.ones((x.shape[0], 1))], 1)

        activation = 1. / (1. + np.exp(-1. * np.matmul(self.W.T, x.T)))
        return activation.T

    def save(self, filename):
        save = np.array({
            'mean': self.mean,
            'std': self.std,
            'weight': self.W,
        })
        np.save(filename, save)

    def load(self, filename):
        params = np.load(filename)
        self.mean = params['mean']
        self.std = params['std']
        self.W = params['weight']

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
    print(df.columns)
    train_x = df.select_dtypes(include=[np.number])

    ohe_hogwarts_house = OneHotEncoder()
    hogwarts_house = ohe_hogwarts_house.fit_transform(df[['Hogwarts House']]).toarray()
    train_y = pd.DataFrame(hogwarts_house, columns=ohe_hogwarts_house.categories_)

    model = LogisticRegression()
    model.fit(train_x, train_y)

    model.save('params.npy')
    np.save('categories', ohe_hogwarts_house.categories_[0])
