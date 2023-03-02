import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import logreg

if __name__ == '__main__':
    np.random.seed(43)

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', default='datasets/dataset_train.csv')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=None,
                        help='1: Stochastic GD, 2 ~ n_samples - 1: Mini Batch GD, n_samples or default: Batch GD')
    args = parser.parse_args()

    df = pd.DataFrame()
    try:
        df = pd.read_csv(args.filepath)
    except FileNotFoundError:
        print(
            f'No such file or directory: \'{args.filepath}\'', file=sys.stderr)
        exit(1)

    train_x = df.drop(columns=['Index']).select_dtypes(include=[np.number])

    ohe_hogwarts_house = OneHotEncoder()
    hogwarts_house = ohe_hogwarts_house.fit_transform(
        df[['Hogwarts House']]).toarray()
    train_y = pd.DataFrame(
        hogwarts_house, columns=ohe_hogwarts_house.categories_)

    model = logreg.LogisticRegression()
    model.fit(train_x, train_y, batch_size=args.batch_size)

    model.save('params.npy')
    np.save('categories', ohe_hogwarts_house.categories_[0])
