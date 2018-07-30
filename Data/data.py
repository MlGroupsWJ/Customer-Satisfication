import pandas as pd

train_data = pd.read_csv('../Data/train.csv')
train_data.drop('ID', axis=1, inplace=True)
target1_data = pd.DataFrame(train_data[train_data.TARGET == 1])

if __name__ == '__main__':
    target1_data.to_csv('../csv/train_target1.csv', index=False)