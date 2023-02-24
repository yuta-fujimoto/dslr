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

    size = df_features.shape[1]
    fig, axes = plt.subplots(size, size)
    for y, y_feature in enumerate(df_features.columns):
        for x, x_feature in enumerate(df_features.columns):
            axes[y][x].scatter(x=df_features[x_feature], y=df_features[y_feature], s=8)
            if y == 0:
                axes[y][x].set_title(x_feature, fontsize=30)
            if x == 0:
                axes[y][x].set_ylabel(y_feature, fontsize=30)

    fig.set_figheight(80)
    fig.set_figwidth(80)
    plt.tight_layout()
    plt.savefig('scatter_plot.png')
    # plt.show()
