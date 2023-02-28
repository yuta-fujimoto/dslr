from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    train_file = 'datasets/dataset_train.csv'
    test_file = 'datasets/dataset_test.csv'
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    train_x = train_df.drop(columns=['Index']).select_dtypes(include=[np.number])
    train_x = train_x.fillna(train_x.mean())

    le_hogwarts_house = LabelEncoder()
    train_y = le_hogwarts_house.fit_transform(
        train_df['Hogwarts House'])

    LR = LogisticRegression(random_state=42, multi_class='ovr')
    LR.fit(X=train_x, y=train_y)
    accuracy = accuracy_score(LR.predict(train_x), train_y)
    print('accuracy: ', accuracy)

    test_index = test_df['Index']
    test_x = test_df.drop(columns=['Index', 'Hogwarts House']).select_dtypes(include=[np.number])
    # fill na with mean of train data
    test_x = test_x.fillna(train_x.mean())

    preds = LR.predict(test_x)
    preds_csv = pd.DataFrame({'Index': test_index, 'Hogwarts House': [le_hogwarts_house.classes_[pred] for pred in preds] })
    preds_csv.to_csv('houses_sklearn.csv', index=False)
