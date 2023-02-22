import numpy as np
import pandas as pd
import argparse
from pprint import pprint

# exclude nan values


def Count(df):
    counts = {}

    for column_name, item in df.iteritems():
        count = 0
        for i in item:
            if not pd.isnull(i):
                count += 1
        counts[column_name] = count

    return counts


def Mean(df):
    means = {}

    counts = Count(df)
    for column_name, item in df.iteritems():
        sum = 0
        for i in item:
            if not pd.isnull(i):
                sum += i
        means[column_name] = sum / counts[column_name]

    return means


def Std(df):
    stds = {}

    counts = Count(df)
    means = Mean(df)
    for column_name, item in df.iteritems():
        variance = 0
        for i in item:
            if not pd.isnull(i):
                variance += (i - means[column_name]) ** 2

        variance /= counts[column_name]
        stds[column_name] = np.sqrt(variance)

    return stds


def Percentile(df, p):
    percentiles = {}

    counts = Count(df)
    for column_name, item in df.iteritems():
        sorted_item = item.sort_values().reset_index(drop=True)
        take = (counts[column_name] - 1) * p
        # linear interpolation
        percentiles[column_name] = sorted_item[np.floor(
            take)] + (sorted_item[np.ceil(take)] - sorted_item[np.floor(take)]) * (take - np.floor(take))

    return percentiles


def extractFeatures(df):
    features = pd.DataFrame(columns=df.columns, index=[
        'Count',
        'Mean',
        'Std',
        'Min',
        '25%',
        '50%',
        '75%',
        'Max'
    ])

    features.loc['Count'] = Count(df)
    features.loc['Mean'] = Mean(df)
    features.loc['Std'] = Std(df)
    features.loc['Min'] = Percentile(df, 0.)
    features.loc['25%'] = Percentile(df, 0.25)
    features.loc['50%'] = Percentile(df, 0.5)
    features.loc['75%'] = Percentile(df, 0.75)
    features.loc['Max'] = Percentile(df, 1.)

    return features

pd.options.display.max_colwidth = 30

parser = argparse.ArgumentParser()
parser.add_argument('filepath')

df = pd.read_csv('datasets/dataset_train.csv')

df = df.select_dtypes(include=[np.number]).drop(columns=['Index'])

features = extractFeatures(df)

print(features)
# print(features.to_string())
# debug
pprint(df.describe())
