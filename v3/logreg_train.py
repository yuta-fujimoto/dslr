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

    def fit(self, x_df, y_df, learning_rate = 0.01, epoch = 10, batch_size = None):
        x = x_df.values
        y = y_df.values

        n_features = x.shape[1]
        n_samples = x.shape[0]
        n_labels = y.shape[1]

        # normalize to optimize learning and prevent overflow 
        self.mean = x_df.mean().values
        self.std = x_df.std().values
        x = (x - self.mean) / self.std
        # concatenate bias
        x = np.concatenate([x, np.ones((n_samples, 1))], 1)

        # Batch GD
        if batch_size == None:
            split = 1
        else: # mini-batch GD or stochastic GD
            split = n_samples // batch_size

        self.W = np.zeros((n_features + 1, n_labels))

        for i in range(epoch):
            batch_size = 0
            loss = np.zeros(n_labels)
            correct = 0
            for x_batch, y_batch in zip(np.array_split(x, split, 0), np.array_split(y, split, 0)):
                activation = 1. / (1. + np.exp(-1. * np.matmul(self.W.T, x_batch.T)))

                loss_batch = -np.diag(np.matmul(y_batch.T, np.log(activation.T)) +
                                 np.matmul(1 - y_batch.T, np.log(1 - activation.T)))
                loss += loss_batch / n_samples

                grad = np.matmul((activation - y_batch.T), x_batch) / n_samples

                self.W = self.W - grad.T * learning_rate

                correct += (activation.T.argmax(1) == y_batch.argmax(1)).sum()
            print(f'epoch {i + 1}:\n loss: {loss} acc: {correct / n_samples:.5}')

    def predict(self, x_df):
        x = x_df.values

        # concatenate bias
        x = np.concatenate([x, np.ones((x.shape[0], 1))], 1)

        activation = 1. / (1. + np.exp(-1. * np.matmul(self.W.T, x.T)))
        return activation.T

    def save(self, filename):
        save = np.array({
            'mean': self.mean,
            "std": self.std,
            "weight": self.W
        })
        np.save(filename, save)

    def load(self, filename):
        params = np.load(filename, allow_pickle=True).item()
        self.mean = params['mean']
        self.std = params['std']
        self.W = params['weight']

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=None,
                         help='1: Stochastic GD, 2 ~ n_samples - 1: Mini Batch GD, n_samples or default: Batch GD')
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

    model = LogisticRegression()
    model.fit(train_x, train_y, batch_size=args.batch_size)

    model.save('params.npy')
    np.save('categories', ohe_hogwarts_house.categories_[0])
