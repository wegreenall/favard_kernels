import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import os
import re
from termcolor import colored
from typing import Tuple

"""
This file contains code that will read the data of the experiment_11_data/single_training,
comparing scores between the cases: - Gaussian input, Gaussian basis; non-Gaussian input, Gaussian basis; non-Gaussian input, Favard basis.
"""


def read_files(
    number: int, prefix: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gaussian_file = "exp_11_gaussian_parameters_" + str(number) + ".pt"
    non_gaussian_file = "exp_11_non_gaussian_parameters_" + str(number) + ".pt"
    favard_file = "exp_11_favard_parameters_" + str(number) + ".pt"
    try:
        gaussian_data = torch.load(prefix + gaussian_file).unsqueeze(0)
        non_gaussian_data = torch.load(prefix + non_gaussian_file).unsqueeze(0)
        favard_data = torch.load(prefix + favard_file).unsqueeze(0)
    except FileNotFoundError:
        print("File not found for number {}".format(number))
        gaussian_data = None
        non_gaussian_data = None
        favard_data = None

    return (gaussian_data, non_gaussian_data, favard_data)


def get_experiment_numbers(prefix: str):
    files = os.listdir(prefix)
    numbers = set()
    number_regex = re.compile(r"(?P<number>[0-9]*).pt")
    for file in files:
        # get the number in the filename
        number_matches = number_regex.search(file)
        try:
            number = number_matches.group("number")
            numbers.add(int(number))
        except AttributeError:
            print(file + " does not contain any numbers!")
    return numbers


if __name__ == "__main__":
    """
    The program begins here
    """
    gaussian_test = True

    if gaussian_test:
        prefix = "/home/william/phd/programming_projects/favard_kernels/experiments/experiment_11_data/single_training/"
    else:
        prefix = "/home/william/phd/programming_projects/favard_kernels/experiments/experiment_11_data/single_training/non_gaussian_test_input_dist/"
    numbers = get_experiment_numbers(prefix)
    experiment_count = len(numbers)
    successes = 0
    for number in numbers:
        gaussian_data, non_gaussian_data, favard_data = read_files(number, prefix)
        print(colored(gaussian_data, "blue"))
        print(colored(non_gaussian_data, "yellow"))
        print(colored(favard_data, "magenta"))
        if gaussian_data is not None:
            gaussian_sum = sum(gaussian_data).item()
            non_gaussian_sum = sum(non_gaussian_data).item()
            favard_sum = sum(favard_data.data).item()
            print(
                "######",
                "\n",
                colored(gaussian_sum, "blue"),
                "\n",
                colored(non_gaussian_sum, "yellow"),
                "\n",
                colored(favard_sum, "magenta"),
                "\n",
                "######",
            )
            if favard_sum > non_gaussian_sum:
                print(colored("Favard is better on number {}".format(number), "green"))
                successes += 1
            else:
                print(
                    colored("Favard is not better on number {}".format(number), "red")
                )
    print("\n ####### \n")
    print("Experiments:", colored(experiment_count, "blue"))
    print("Successes:", colored(successes, "cyan"))
    print(
        "Success rate:",
        colored(
            successes / experiment_count,
            "green" if successes / experiment_count > 0.5 else "red",
        ),
    )
    print("\n ####### \n")
