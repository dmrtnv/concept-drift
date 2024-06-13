import numpy as np
import matplotlib.pyplot as plt

from src.params import periods

PERIODS = periods.PERIODS[:-1]


class Graph:
    def __init__(self, values, color, marker, linestyle, label) -> None:
        self.values = values
        self.avg_value = round(sum(self.values) / len(self.values), 1)
        self.color = color
        self.marker = marker
        self.linestyle = linestyle
        self.label = label


class Plot:
    def __init__(self, title, picture_name) -> None:
        self.title = title
        self.picture_name = picture_name
        self.graphs = []

    def add_graph(self, graph):
        self.graphs.append(graph)

    def display(self):
        for graph in self.graphs:
            plt.plot(
                PERIODS,
                graph.values,
                color=graph.color,
                marker=graph.marker,
                linestyle=graph.linestyle,
                label=graph.label,
            )
            plt.axhline(
                y=graph.avg_value,
                color=graph.color,
                linestyle="dashed",
                linewidth=1,
                label=f"{graph.label} avg ({graph.avg_value}%)",
            )

        plt.legend()
        plt.title(self.title)

        plt.xlabel("Time Period")
        plt.ylabel("Percentage %")

        plt.xticks(rotation=70)
        plt.yticks(np.arange(30.0, 100.1, 5))

        plt.grid()

        figure = plt.gcf()
        figure.set_size_inches(12, 6)
        plt.savefig(self.picture_name, bbox_inches="tight", pad_inches=0.3, dpi=200)
        plt.show()
