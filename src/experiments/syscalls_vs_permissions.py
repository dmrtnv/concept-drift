import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.params.features import PERMISSIONS, SYSTEMCALLS
from src.visualization.results import Graph, Plot
from src.utils.evaluate import evaluate
from src.params.periods import PERIODS

FEATURES = PERMISSIONS + SYSTEMCALLS
READ_PATH = './data/descending_realistic'


class Scores:
    def __init__(self, name) -> None:
        self.name = name
        self.accuracy = []
        self.f1 = []
        self.precision = []
        self.recall = []
        self.specificity = []

    def get_scores(self, accuracy, precision, recall, f1, specificity):
        self.accuracy.append(accuracy)
        self.f1.append(f1)
        self.precision.append(precision)
        self.recall.append(recall)
        self.specificity.append(specificity)

    def display(self):
        print(f"{self.name}_accuracy =", self.accuracy)
        print(f"{self.name}_f1 =", self.f1)
        print(f"{self.name}_recall =", self.recall)
        print(f"{self.name}_precision =", self.precision)
        print(f"{self.name}_specificity =", self.specificity)


def train_model(data, balancer=None, classweight=None):
    X = data[FEATURES]
    y = data.Malware

    if balancer != None:
        X, y = balancer.fit_resample(X, y)

    if classweight == "balanced":
        model = RandomForestClassifier(class_weight="balanced")
    else:
        model = RandomForestClassifier(n_jobs=-1)

    model.fit(X, y)

    return model

def descending_realistic_data():
    sys_scores = Scores("sys_scores")
    per_scores = Scores("per_scores")
    both_scores = Scores("both_scores")

    data_train = pd.read_csv(f"{READ_PATH}/train/{PERIODS[0]}.csv")

    sys_model = RandomForestClassifier(n_jobs=-1)
    sys_model.fit(data_train[SYSTEMCALLS], data_train.Malware)

    per_model = RandomForestClassifier(n_jobs=-1)
    per_model.fit(data_train[PERMISSIONS], data_train.Malware)

    both_model = RandomForestClassifier(n_jobs=-1)
    both_model.fit(data_train[FEATURES], data_train.Malware)

    for period in PERIODS:
      train_file = (
          f"{READ_PATH}/train/{period}.csv"
      )
      test_file = (
          f"{READ_PATH}/test/{period}.csv"
      )

      if (
          train_file == f"{READ_PATH}/train/{PERIODS[0]}.csv"
          or test_file
          == f"{READ_PATH}/test/{PERIODS[0]}.csv"
      ):
          continue

      new_data_train = pd.read_csv(train_file)
      data_test = pd.read_csv(test_file)

      # evaluate
      accuracy, precision, recall, f1, specificity = evaluate(
          sys_model, data_test, SYSTEMCALLS
      )
      sys_scores.get_scores(accuracy, precision, recall, f1, specificity)
      print(f"{period}-M1 ", accuracy, precision, recall, f1, specificity)

      accuracy, precision, recall, f1, specificity = evaluate(
          per_model, data_test, PERMISSIONS
      )
      per_scores.get_scores(accuracy, precision, recall, f1, specificity)
      print(f"{period}-M2 ", accuracy, precision, recall, f1, specificity)

      accuracy, precision, recall, f1, specificity = evaluate(
          both_model, data_test, FEATURES
      )
      both_scores.get_scores(accuracy, precision, recall, f1, specificity)
      print(f"{period}-M3 ", accuracy, precision, recall, f1, specificity)

      # retrain

      data_train = pd.concat([data_train, new_data_train]).sample(frac=1)

      sys_model.fit(data_train[SYSTEMCALLS], data_train.Malware)
      per_model.fit(data_train[PERMISSIONS], data_train.Malware)
      both_model.fit(data_train[FEATURES], data_train.Malware)

    sys_scores.display()
    per_scores.display()
    both_scores.display()

    # visualize
    accuracy_plot = Plot(
        "Accuracy comparison of System calls, Permission and Both",
        "descending_realistic_data_accuracy_comparison_systemcalls_permissions_both.png",
    )
    accuracy_plot.add_graph(Graph(sys_scores.accuracy, "c", "o", "-", "System calls"))
    accuracy_plot.add_graph(Graph(per_scores.accuracy, "y", "s", "-.", "Permissions"))
    accuracy_plot.add_graph(
        Graph(both_scores.accuracy, "m", "v", ":", "System calls + Permissions")
    )
    # accuracy_plot.add_graph(Graph(sgd_incremental_scores.accuracy, 'g', '*', '--', 'SGD incremental learning'))
    accuracy_plot.display()

    f1_plot = Plot(
        "F1 comparison of System calls, Permission and Both",
        "descending_realistic_data_f1_comparison_systemcalls_permissions_both.png",
    )
    f1_plot.add_graph(Graph(sys_scores.f1, "c", "o", "-", "System calls"))
    f1_plot.add_graph(Graph(per_scores.f1, "y", "s", "-.", "Permissions"))
    f1_plot.add_graph(
        Graph(both_scores.f1, "m", "v", ":", "System calls + Permissions")
    )
    f1_plot.display()

    recall_plot = Plot(
        "Recall comparison of System calls, Permission and Both",
        "descending_realistic_data_recall_comparison_systemcalls_permissions_both.png",
    )
    recall_plot.add_graph(Graph(sys_scores.recall, "c", "o", "-", "System calls"))
    recall_plot.add_graph(Graph(per_scores.recall, "y", "s", "-.", "Permissions"))
    recall_plot.add_graph(
        Graph(both_scores.recall, "m", "v", ":", "System calls + Permissions")
    )
    recall_plot.display()

    precision_plot = Plot(
        "Precision comparison of System calls, Permission and Both",
        "descending_realistic_data_precision_comparison_systemcalls_permissions_both.png",
    )
    precision_plot.add_graph(Graph(sys_scores.precision, "c", "o", "-", "System calls"))
    precision_plot.add_graph(Graph(per_scores.precision, "y", "s", "-.", "Permissions"))
    precision_plot.add_graph(
        Graph(both_scores.precision, "m", "v", ":", "System calls + Permissions")
    )
    precision_plot.display()

    specificity_plot = Plot(
        "Specificity comparison of System calls, Permission and Both",
        "descending_realistic_data_specificity_comparison_systemcalls_permissions_both.png",
    )
    specificity_plot.add_graph(
        Graph(sys_scores.specificity, "c", "o", "-", "System calls")
    )
    specificity_plot.add_graph(
        Graph(per_scores.specificity, "y", "s", "-.", "Permissions")
    )
    specificity_plot.add_graph(
        Graph(both_scores.specificity, "m", "v", ":", "System calls + Permissions")
    )
    specificity_plot.display()


descending_realistic_data()