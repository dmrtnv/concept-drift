import pandas as pd
from src.params import features, periods
from src.utils.split_data import split_data

READ_PATH = './data/periodized'
WRITE_PATH = './data/descending_realistic'

PERIODS = periods.PERIODS
FEATURES = features.PERMISSIONS + features.SYSTEMCALLS
TEST_SIZE = 0.2

PORTION_OF_NEW = 0.15


def main():
  data_test = pd.DataFrame(columns=FEATURES)

  print('hello')

  for period in PERIODS:
    file_read_path = f'{READ_PATH}/{period}.csv'
    file_write_train_path = f'{WRITE_PATH}/train/{period}.csv'
    file_write_test_path = f'{WRITE_PATH}/test/{period}.csv'
    print(f'reading from {file_read_path}...')

    data = pd.read_csv(file_read_path)

    new_data_train, new_data_test = split_data(data, TEST_SIZE)

    number_of_old_samples = round((1 - PORTION_OF_NEW) * len(data_test))

    if data_test.shape[0]:
      data_test = pd.concat([new_data_test,  data_test.sample(n=number_of_old_samples, replace=True)], axis=0)
    else:
      data_test = pd.concat([new_data_test], axis=0)
    
    print(f'writing to {file_write_train_path}...')
    new_data_train.to_csv(file_write_train_path, index=False)
    
    print(f'writing to {file_write_test_path}...')
    data_test.to_csv(file_write_test_path, index=False)


main()