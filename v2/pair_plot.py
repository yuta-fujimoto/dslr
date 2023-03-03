import numpy as np
import pandas as pd
import argparse
import sys
import seaborn as sns

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

    # exclude all nan columns(test csv) and Index
    drop_columns = df.columns[df.isnull().all().values].tolist()
    drop_columns.append('Index')
    df_features = df.select_dtypes(include=[np.number]).drop(columns=drop_columns)

    # add Hogwarts House as hue
    df_features = pd.concat([df_features, df['Hogwarts House']], axis=1)

    pg = sns.pairplot(df_features, hue='Hogwarts House')
    pg.savefig('pair_plot.png')
