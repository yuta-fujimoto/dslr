import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import sys

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

    # exclude all nan columns(test csv)
    drop_columns = df.columns[df.isnull().all().values].tolist()
    drop_columns.append('Index')
    df_features = df.select_dtypes(include=[np.number]).drop(columns=drop_columns)

    df_target = df['Hogwarts House']
    labels = df_target.unique()

    # plot
    size = int(np.ceil(df_features.shape[1] ** 0.5))
    fig, axes = plt.subplots(size, size)
    for i, column in enumerate(df_features.columns):
        axes[i // size][i % size].set_title(column, fontsize=8)
        for label in labels:
            x = df_features.loc[df_target == label, column]
            axes[i // size][i % size].hist(x, label=label, alpha=0.6, bins=30)

    # delete exceed subplots
    for i in range(len(df_features.columns), size * size):
        axes[i // size][i % size].axis('off')

    fig.legend(labels=labels, loc='lower right')
    plt.tight_layout()
    plt.savefig('histogram.png')
    # plt.show()
