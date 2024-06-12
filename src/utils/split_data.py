from sklearn.model_selection import train_test_split

def split_data(data, features, testsize):
  X = data[features]
  y = data.Malware

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize)

  X_train["Malware"] = y_train
  X_test["Malware"] = y_test

  return X_train, X_test