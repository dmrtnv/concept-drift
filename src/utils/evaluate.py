from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

def evaluate(model, data, features):
  X_test = data[features]
  y_test = data.Malware

  y_pred = model.predict(X_test)

  accuracy = round(accuracy_score(y_test, y_pred) * 100, 1)
  precision = round(precision_score(y_test, y_pred) * 100, 1)
  recall = round(recall_score(y_test, y_pred) * 100, 1)
  f1 = round(f1_score(y_test, y_pred) * 100, 1)
  tn, fp, *_ = confusion_matrix(y_test, y_pred).ravel()
  specificity = round(100 * tn / (tn + fp), 1)

  return accuracy, precision, recall, f1, specificity