import torch
import numpy as np
import pandas as pd
from scipy.stats import kstest
from enum import Enum
import latextable
import texttable as tt


class DataSet(Enum):
    RED = 1
    WHITE = 2
    BOTH = 3


def standardise_data(data):
    return (data - np.mean(data)) / np.std(data)


def get_data(type: DataSet, standardise=True):
    """
    Returns the data for the wine quality analysis.
    """
    if type == DataSet.RED:
        # red wine
        red_wine_data = pd.read_csv("winequality-red.csv", sep=";")
        free_sulphur = torch.Tensor(
            red_wine_data["free sulfur dioxide"]
        ).numpy()
        total_sulphur = torch.Tensor(
            red_wine_data["total sulfur dioxide"]
        ).numpy()

    elif type == DataSet.WHITE:
        white_wine_data = pd.read_csv("winequality-white.csv", sep=";")
        free_sulphur = torch.Tensor(
            white_wine_data["free sulfur dioxide"]
        ).numpy()
        total_sulphur = torch.Tensor(
            white_wine_data["total sulfur dioxide"]
        ).numpy()

    elif type == DataSet.BOTH:
        # concatenate the two datasets
        red_wine_data = pd.read_csv("winequality-red.csv", sep=";")
        white_wine_data = pd.read_csv("winequality-white.csv", sep=";")
        free_sulphur = torch.Tensor(
            pd.concat(
                (
                    red_wine_data["free sulfur dioxide"],
                    white_wine_data["free sulfur dioxide"],
                ),
                ignore_index=True,
            )
        ).numpy()
        total_sulphur = torch.Tensor(
            pd.concat(
                (
                    red_wine_data["total sulfur dioxide"],
                    white_wine_data["total sulfur dioxide"],
                ),
                ignore_index=True,
            )
        ).numpy()
    if standardise:
        free_sulphur = standardise_data(free_sulphur)
        total_sulphur = standardise_data(total_sulphur)
    return free_sulphur, total_sulphur


if __name__ == "__main__":
    datasets = [DataSet.RED, DataSet.WHITE, DataSet.BOTH]
    # perform a kolmogorov-smirnov test on the data
    stats = []
    p_values = []
    for dataset in datasets:
        _, data = get_data(dataset, standardise=True)
        # perform a kolmogorov-smirnov test on the data, with respect to the normal cdf
        ks_test = kstest(data, "norm")
        stats.append(ks_test[0])
        p_values.append(ks_test[1])
    print(stats)
    print(p_values)
    table = tt.Texttable()
    table.set_precision(10)
    table.set_cols_align(["c", "c", "c"])
    table.set_cols_valign(["m", "m", "m"])
    table.add_rows(
        [
            ["Dataset", "KS Statistic", "p-value"],
            ["Red Wine", stats[0], p_values[0]],
            ["White Wine", stats[1], p_values[1]],
            ["Amalgamated Dataset", stats[2], p_values[2]],
        ]
    )
    table_tex = latextable.draw_latex(
        table, caption="KS Test Results", label="tab:wine_ks_test"
    )
    # write table tex to file
    with open("wine_dataset_ks.tex", "w") as f:
        f.write(table_tex)

    # print(ks_test)
