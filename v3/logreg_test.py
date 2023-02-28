import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logreg

if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath')
    args = parser.parse_args()

    df = pd.DataFrame()
    try:
        df = pd.read_csv(args.filepath)
    except FileNotFoundError:
        print(
            f'No such file or directory: \'{args.filepath}\'', file=sys.stderr)
        exit(1)

    df_index = df['Index']
    df = df.drop(columns=['Index']).reset_index(drop=True)
    test = df.select_dtypes(include=[np.number])
    test = test.drop(columns=['Hogwarts House'])

    model = logreg.LogisticRegression()
    model.load('params.npy')
    categories = np.load('categories.npy', allow_pickle=True)

    preds_prob = model.predict(test)
    preds = preds_prob.argmax(1)

    result = pd.DataFrame({'Index': df_index, 'Hogwarts House': [
                          categories[p] for p in preds]})
    result.to_csv('houses.csv', index=False)
