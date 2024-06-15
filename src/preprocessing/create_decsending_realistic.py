import pandas as pd
from src.params import features, periods
from src.utils.split_data import split_data

READ_PATH = './data/periodized'
WRITE_PATH = './data/descending_realistic'

PERIODS = periods.PERIODS
FEATURES = features.PERMISSIONS + features.SYSTEMCALLS
TEST_SIZE = 0.2
PORTION_OF_NEW = 0.15
MAX_NUMBER_OF_NEW_SAMPLES=round(PORTION_OF_NEW * 4000)


def scale_down_test(data: pd.DataFrame, max_values: int, min_malware_portion: float = 0):
  data_benign = data[data['Malware'] == 0].reset_index(drop=True)
  data_malicious = data[data['Malware'] == 1].reset_index(drop=True)

  if data_benign.shape[0] < max_values * min_malware_portion:
    data_malicious = data_malicious.sample(n=(max_values - data_benign.shape[0]))

    return pd.concat([data_benign, data_malicious], axis=0).sample(frac=1)
  
  return data.sample(n=max_values)


def main():
  data_test = pd.DataFrame(columns=FEATURES)

  for period in PERIODS:
    print(f'========================= {period} =========================')

    file_read_path = f'{READ_PATH}/{period}.csv'
    file_write_train_path = f'{WRITE_PATH}/train/{period}.csv'
    file_write_test_path = f'{WRITE_PATH}/test/{period}.csv'
    print(f'reading from {file_read_path}...')

    data = pd.read_csv(file_read_path)

    print(f'Original data: benign: {data.Malware.value_counts().get(0)}; malicious {data.Malware.value_counts().get(1)}; total: {data.shape[0]}.')

    new_data_train, new_data_test = split_data(data, FEATURES, TEST_SIZE)

    print(f'Original new_data_test: benign: {new_data_test.Malware.value_counts().get(0)}; malicious {new_data_test.Malware.value_counts().get(1)}; total: {new_data_test.shape[0]}.')

    if new_data_test.shape[0] > MAX_NUMBER_OF_NEW_SAMPLES:
      new_data_test = scale_down_test(new_data_test, MAX_NUMBER_OF_NEW_SAMPLES, min_malware_portion=0.1)
      print(f'Scaled new_data_test: benign: {new_data_test.Malware.value_counts().get(0)}; malicious {new_data_test.Malware.value_counts().get(1)}; total: {new_data_test.shape[0]}.')

    number_of_old_samples = round(((1 - PORTION_OF_NEW) / PORTION_OF_NEW) * new_data_test.shape[0])

    if data_test.shape[0]:
      data_test = pd.concat([new_data_test,  data_test.sample(n=number_of_old_samples, replace=True)], axis=0)
    else:
      data_test = pd.concat([new_data_test], axis=0)

    print(f'data_test: benign: {data_test.Malware.value_counts().get(0)}; malicious {data_test.Malware.value_counts().get(1)}; total: {data_test.shape[0]}.')
    
    print(f'writing to {file_write_train_path}...')
    new_data_train.to_csv(file_write_train_path, index=False)
    
    print(f'writing to {file_write_test_path}...')
    data_test.to_csv(file_write_test_path, index=False)

    print(f'===========================================================\n')

main()