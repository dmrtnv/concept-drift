import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from src.params import periods

PERIODS = periods.PERIODS[:-1]

def max_sum_of_pairs(arr1, arr2):
    max_sum = float('-inf')

    for i in range(len(arr1)):
        current_sum = arr1[i] + arr2[i]
        max_sum = max(max_sum, current_sum)
    
    return max_sum


def display_distribution(benign, malicious):
    _, ax = plt.subplots()

    positions_test = np.arange(len(PERIODS) * 2, 0, -2)

    p_benign = ax.barh(
        positions_test,
        benign,
        height=1.8,
        label="benign",
        color="#025aad",
        edgecolor="white",
    )
    p_malicious = ax.barh(
        positions_test,
        malicious,
        height=1.8,
        left=benign,
        label="malicious",
        color="#702626",
        edgecolor="white",
    )
    ax.bar_label(
        p_malicious,
        labels=[f"{b} / {m}" for b, m in zip(benign, malicious)],
        padding=6,
        fontsize=10,
    )

    plt.title(f"Goodware / Malware distribution in descending realistic test datasets")

    plt.ylabel("Time Period")
    plt.xlabel("Number of samples")

    plt.yticks([r for r in range(len(PERIODS) * 2, 0, -2)], PERIODS)

    step = 1000
    max_range = max_sum_of_pairs(benign, malicious)

    max_range = (math.ceil(max_range / step) + 1) * step + 1

    print(max_range)

    plt.xticks(np.arange(0, max_range, step))
    plt.legend()

    figure = plt.gcf()
    figure.set_size_inches(12, 9)
    plt.savefig(
        "./plots/distribution-descending-realistic.png",
        bbox_inches="tight",
        pad_inches=0.3,
        dpi=200,
    )
    plt.show()


def main():
    benign = []
    malicious = []

    for period in PERIODS:
        file = f"./data/descending_realistic/test/{period}.csv"
        data = pd.read_csv(file)

        number_of_benign = data.Malware.value_counts().get(0, 0)
        number_of_malicious = data.Malware.value_counts().get(1, 0)

        print(f'{period}: b {number_of_benign}, m {number_of_malicious}')


        benign.append(number_of_benign)
        malicious.append(number_of_malicious)

    display_distribution(benign, malicious)

main()