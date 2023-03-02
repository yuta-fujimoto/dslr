import numpy as np
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, W=None) -> None:
        # weight and bias
        self.W = W
        self.mean = 0.
        self.std = 0.
        self.features = [
            'Astronomy',
            'Herbology',
            'Divination',
            'Muggle Studies',
            'Ancient Runes',
            'Transfiguration',
            'Charms',
            'History of Magic',
        ]

    def fit(self, x_df, y_df, learning_rate=0.001, epoch=10, batch_size=None):
        y = y_df.values

        x_df = x_df[self.features]
        self.mean = x_df.mean()
        self.std = x_df.std()

        x = x_df.fillna(self.mean).values
        # fillna and normalization
        # normalize to optimize learning and prevent overflow
        x = (x - self.mean.values) / self.std.values

        n_features = x.shape[1]
        n_samples = x.shape[0]
        n_labels = y.shape[1]

        # concatenate bias
        x = np.concatenate([x, np.ones((n_samples, 1))], 1)

        # Batch GD
        if batch_size == None:
            split = 1
        else:  # mini-batch GD or stochastic GD
            split = n_samples // batch_size

        self.W = np.zeros((n_features + 1, n_labels))

        for i in range(epoch):
            train_x, valid_x, train_y, valid_y = train_test_split(
                x, y, test_size=0.2, shuffle=True)
            log = {}

            # train
            correct = 0
            loss = np.zeros(n_labels)
            for x_batch, y_batch in zip(np.array_split(train_x, split, 0), np.array_split(train_y, split, 0)):
                activation = 1. / \
                    (1. + np.exp(-1. * np.matmul(self.W.T, x_batch.T)))
                loss_batch = -np.diag(np.matmul(y_batch.T, np.log(activation.T)) +
                                      np.matmul(1 - y_batch.T, np.log(1 - activation.T)))
                loss += loss_batch / train_x.shape[0]

                grad = np.matmul((activation - y_batch.T), x_batch) / n_samples

                self.W = self.W - grad.T * learning_rate
                correct += (activation.T.argmax(1) == y_batch.argmax(1)).sum()
            log['loss'] = loss
            log['acc'] = correct / train_x.shape[0]

            # test
            correct = 0
            loss = np.zeros(n_labels)
            for x_batch, y_batch in zip(np.array_split(valid_x, 1, 0), np.array_split(valid_y, 1, 0)):
                activation = 1. / \
                    (1. + np.exp(-1. * np.matmul(self.W.T, x_batch.T)))
                loss_batch = -np.diag(np.matmul(y_batch.T, np.log(activation.T)) +
                                      np.matmul(1 - y_batch.T, np.log(1 - activation.T)))
                loss += loss_batch / valid_x.shape[0]

                correct += (activation.T.argmax(1) == y_batch.argmax(1)).sum()
            log['val_loss'] = loss
            log['val_acc'] = correct / valid_x.shape[0]

            print(
                f'epoch {i + 1}/{epoch}: - loss: {log["loss"]} - acc: {log["acc"]:.4} - val_loss: {log["val_loss"]} - val_acc: {log["val_acc"]:.4}')

    def predict(self, x_df):
        x_df = x_df[self.features]

        # fillna and normalization
        x = x_df.fillna(self.mean).values
        x = (x - self.mean.values) / self.std.values

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
