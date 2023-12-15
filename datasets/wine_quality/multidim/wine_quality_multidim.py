import torch
import torch.distributions as D
import pandas as pd
import matplotlib.pyplot as plt

from enum import Enum
from mercergp.builders import (
    build_mercer_gp,
    build_mercer_gp_fourier_posterior,
    train_mercer_params,
    build_smooth_exponential_mercer_gp,
    build_smooth_exponential_mercer_gp_fourier_posterior,
    train_smooth_exponential_mercer_params,
)
import ortho.basis_functions as bf
from mercergp.MGP import MercerGP, MercerGPFourierPosterior
from mercergp.kernels import MercerKernel
from mercergp.eigenvalue_gen import SmoothExponentialFasshauer
from mercergp.likelihood_refit import Likelihood
from ortho.builders import (
    OrthoBuilder,
    get_orthonormal_basis_from_sample_multidim,
)
from termcolor import colored
import pickle
from typing import List, Tuple
import random
import tikzplotlib

from favard_kernels.datasets.wine_quality.wine_quality_analysis_2 import (
    GPType,
    DataSet,
    KernelType,
    get_data,
)


def get_GP_multidim(
    order,
    trained_parameters,
    trained_noise,
    input_sample,
    output_sample,
    gp_type,
    kernel_type,
    dimension=2,
):
    """
    Returns a GP model.
    """
    trained_parameters[0]["noise_parameter"] = trained_noise
    if kernel_type == KernelType.MERCER:
        basis = bf.Basis(
            [bf.smooth_exponential_basis_fasshauer] * dimension,
            dimension,
            order,
            trained_parameters,
        )
    elif kernel_type == KernelType.FAVARD:
        # get the basis from the data
        weight_functions = [weight_function] * dimension
        basis = get_orthonormal_basis_from_sample_multidim(
            input_sample, weight_functions, order, parameters=[{}, {}]
        )
        # # contour plot
        values = basis(input_sample)
        x_axis = torch.linspace(-3, 3, 100)
        y_axis = torch.linspace(-3, 3, 100)
        X, Y = torch.meshgrid(x_axis, y_axis)
        Z = torch.stack((X, Y), dim=2)
        Z_reshaped = Z.reshape(-1, 2)
        output = basis(Z_reshaped).reshape(100, 100, -1)[:, :, 2]
        plt.contourf(X, Y, output)
        plt.show()

    eigenvalue_generator = SmoothExponentialFasshauer(order, dimension)
    eigenvalues = eigenvalue_generator(trained_parameters)

    kernel = MercerKernel(order, basis, eigenvalues, trained_parameters)
    if kernel_type == KernelType.MERCER:
        if gp_type == GPType.STANDARD:
            gp = build_mercer_gp(
                trained_parameters,
                order,
                basis,
                eigenvalue_generator,
                dim=2,
            )
        elif gp_type == GPType.FOURIER:
            gp = build_smooth_exponential_mercer_gp_fourier_posterior(
                trained_parameters,
                order,
                dim=2,
                begin=-25,
                end=25,
                frequency=5,
                rff_order=4000,
            )

    elif kernel_type == KernelType.FAVARD:
        smooth_exponential_eigenvalues = SmoothExponentialFasshauer(order)
        if gp_type == GPType.STANDARD:
            gp = build_mercer_gp(
                trained_parameters, order, basis, eigenvalue_generator, dim=2
            )
        elif gp_type == GPType.FOURIER:
            gp = build_mercer_gp_fourier_posterior(
                trained_parameters,
                order,
                basis,
                smooth_exponential_eigenvalues,
                begin=-25,
                end=25,
                frequency=5,
                rff_order=4000,
                # prior_variance=4.0,
            )
    # breakpoint()
    gp.add_data(input_sample, output_sample)
    return gp


def get_randomized_data(
    input_sample, output_sample, test_input_count, input_count
):
    """
    Selects test_input_count, input_count values from the input and output data and returns them.

    """
    # to ensure that there are not repeated indices, we sample without
    # replacement the whole set. Then, split into pieces.
    sampled_inputs: List[int] = random.sample(
        range(len(total_sulphur)),
        test_random_sample_point_count + input_random_sample_point_count,
    )
    input_point_indices = sampled_inputs[test_random_sample_point_count:]
    test_point_indices = sampled_inputs[:test_random_sample_point_count]

    # now pick out the data
    input_points = torch.stack(
        [
            free_sulphur[input_point_indices],
            total_sulphur[input_point_indices],
        ],
        dim=1,
    )
    output_points = output_sample[input_point_indices]

    test_input_points = torch.stack(
        [
            free_sulphur[test_point_indices],
            total_sulphur[test_point_indices],
        ],
        dim=1,
    )
    test_output_points = output_sample[test_point_indices]
    return test_input_points, test_output_points, input_points, output_points


def compare_gps(
    favard_gp,
    mercer_gp,
    input_sample,  # the inputs
    output_sample,  # the outputs
    empirical_experiment_count,
    test_input_count,
    input_count,
    # random_sample_point_count,
) -> Tuple[list, list]:
    """
    Now that we have the parameters for the Gaussian processes,
    compare the two using predictive densities.
    """
    favard_predictive_densities = []
    mercer_predictive_densities = []

    for i in range(empirical_experiment_count):
        if i % 100 == 0:
            print("Iteration:", i)

        """
        At each iteration, we produce a random sample of indices
        for the _inputs_, and a random sample of indices from
        for the _test inputs_, that is not the same
        """
        (
            test_input_sample,
            test_output_sample,
            inputs,
            outputs,
        ) = get_randomized_data(
            input_sample, output_sample, test_input_count, input_count
        )
        # set the gps to have these data
        favard_gp.set_data(inputs, outputs)
        mercer_gp.set_data(inputs, outputs)

        # get the predictive densities
        mercer_predictive_density = mercer_gp.get_predictive_density(
            test_input_sample
        )
        favard_predictive_density = favard_gp.get_predictive_density(
            test_input_sample
        )
        # get the predictive density values
        mercer_predictive_density_values = mercer_predictive_density.log_prob(
            test_output_sample
        )
        favard_predictive_density_values = favard_predictive_density.log_prob(
            test_output_sample
        )
        # plt.scatter(
        # input_sample[random_indices], output_sample[random_indices]
        # )
        # plt.show()
        print(
            colored("Favard is better", "green")
            if favard_predictive_density_values
            > mercer_predictive_density_values
            else colored("Mercer is better", "red")
        )
        print(
            "favard log predictive density:",
            favard_predictive_density_values,
        )
        print(
            "mercer wine log predictive density:",
            mercer_predictive_density_values,
        )
        favard_predictive_densities.append(favard_predictive_density_values)
        mercer_predictive_densities.append(mercer_predictive_density_values)
    return favard_predictive_densities, mercer_predictive_densities


def standardise_data(data):
    return (data - torch.mean(data)) / torch.std(data)


def weight_function(input: torch.Tensor):
    return torch.exp(-(torch.abs(input**3)) / 4)


def get_data_multidim(
    type: DataSet, standardise=True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get the data from the wine quality dataset
    """
    output_variable = "residual sugar"
    if type == DataSet.RED:
        df = pd.read_csv(
            "/home/william/phd/programming_projects/favard_kernels/datasets/wine_quality/winequality-red.csv",
            delimiter=";",
        )
        free_sulphur = torch.Tensor(df["free sulfur dioxide"])
        total_sulphur = torch.Tensor(df["total sulfur dioxide"])
        output = torch.Tensor(df[output_variable])
    elif type == DataSet.WHITE:
        df = pd.read_csv(
            "/home/william/phd/programming_projects/favard_kernels/datasets/wine_quality/winequality-white.csv",
            delimiter=";",
        )
        free_sulphur = torch.Tensor(df["free sulfur dioxide"])
        total_sulphur = torch.Tensor(df["total sulfur dioxide"])
        output = torch.Tensor(df[output_variable])
    elif type == DataSet.BOTH:
        df_red = pd.read_csv(
            "/home/william/phd/programming_projects/favard_kernels/datasets/wine_quality/winequality-red.csv",
            delimiter=";",
        )
        df_white = pd.read_csv(
            "/home/william/phd/programming_projects/favard_kernels/datasets/wine_quality/winequality-white.csv",
            delimiter=";",
        )
        free_sulphur = torch.Tensor(
            pd.concat(
                (
                    df_red["free sulfur dioxide"],
                    df_white["free sulfur dioxide"],
                ),
                ignore_index=True,
            )
        )
        total_sulphur = torch.Tensor(
            pd.concat(
                (
                    df_red["total sulfur dioxide"],
                    df_white["total sulfur dioxide"],
                ),
                ignore_index=True,
            )
        )
        output = torch.Tensor(
            pd.concat(
                (
                    df_red[output_variable],
                    df_white[output_variable],
                ),
                ignore_index=True,
            )
        )

    else:
        raise ValueError("Invalid dataset type")
    if standardise:
        free_sulphur = standardise_data(free_sulphur)
        total_sulphur = standardise_data(total_sulphur)
        # output = standardise_data(output)
    return (total_sulphur, free_sulphur, output)


def get_parameters(
    order, total_sulphur, free_sulphur, output_sample, kernel_type
):
    # initial parameters for the optimisation
    dimension = 2
    initial_noise = torch.Tensor([1.0])
    initial_parameters_dim_1 = {
        "ard_parameter": torch.Tensor([1.0]),
        "precision_parameter": torch.Tensor([1.0]),
        "noise_parameter": initial_noise,
        "variance_parameter": torch.Tensor([1.0]),
    }
    initial_parameters_dim_2 = {
        "ard_parameter": torch.Tensor([1.0]),
        "precision_parameter": torch.Tensor([1.0]),
        "noise_parameter": initial_noise,
        "variance_parameter": torch.Tensor([1.0]),
    }
    initial_parameters = [initial_parameters_dim_1, initial_parameters_dim_2]
    input_sample = torch.stack((total_sulphur, free_sulphur), dim=1)
    if kernel_type == KernelType.MERCER:
        basis = bf.Basis(
            [bf.smooth_exponential_basis_fasshauer] * dimension,
            dimension,
            order,
            initial_parameters,
        )
    elif kernel_type == KernelType.FAVARD:
        # get the basis from the data
        weight_functions = [weight_function] * dimension
        basis = get_orthonormal_basis_from_sample_multidim(
            input_sample, weight_functions, order, parameters=[{}, {}]
        )
        # # contour plot
        values = basis(input_sample)
        x_axis = torch.linspace(-3, 3, 100)
        y_axis = torch.linspace(-3, 3, 100)
        X, Y = torch.meshgrid(x_axis, y_axis)
        Z = torch.stack((X, Y), dim=2)
        Z_reshaped = Z.reshape(-1, 2)
        output = basis(Z_reshaped).reshape(100, 100, -1)[:, :, 0]

    eigenvalue_generator = SmoothExponentialFasshauer(order, dimension)
    eigenvalues = eigenvalue_generator(initial_parameters)

    kernel = MercerKernel(order, basis, eigenvalues, initial_parameters)

    likelihood = Likelihood(
        order,
        kernel,
        input_sample,
        output_sample,
        eigenvalue_generator,
        param_learning_rate=0.0001,
        sigma_learning_rate=0.0001,
    )
    trained_noise, trained_parameters = likelihood.fit(
        initial_noise, initial_parameters, max_iterations=25000
    )
    return trained_parameters, trained_noise


if __name__ == "__main__":
    pretrained = False
    precompared = False
    save_tikz = True
    empirical_experiment_count = 1000
    dataset = DataSet.BOTH
    if dataset == DataSet.RED:
        code = "red"
    elif dataset == DataSet.WHITE:
        code = "white"
    elif dataset == DataSet.BOTH:
        code = "both"
    # inputs

    order = 6
    (total_sulphur, free_sulphur, output_sample) = get_data_multidim(dataset)

    # build the first example where we get the Mercer parameters
    """
    First, get the parameters for this GP - i.e. do the training
    """
    if not pretrained:
        # Mercer
        (
            mercer_trained_parameters,
            mercer_trained_noise,
        ) = get_parameters(
            order,
            total_sulphur,
            free_sulphur,
            output_sample,
            KernelType.MERCER,
        )
        torch.save(
            mercer_trained_noise,
            "./mercer_trained_noise_{}.pt".format(code),
        )
        with open(
            "./mercer_trained_parameters_{}.pkl".format(code),
            "wb",
        ) as f:
            pickle.dump(mercer_trained_parameters, f)

        # Favard
        (
            favard_trained_parameters,
            favard_trained_noise,
        ) = get_parameters(
            order,
            total_sulphur,
            free_sulphur,
            output_sample,
            KernelType.FAVARD,
        )
        torch.save(
            favard_trained_noise,
            "./favard_trained_noise_{}.pt".format(code),
        )
        with open(
            "./favard_trained_parameters_{}.pkl".format(code),
            "wb",
        ) as f:
            pickle.dump(favard_trained_parameters, f)
        with open(
            "./favard_trained_parameters_{}.pkl".format(code),
            "wb",
        ) as f:
            pickle.dump(favard_trained_parameters, f)
        with open(
            "./mercer_trained_parameters_{}.pkl".format(code),
            "wb",
        ) as f:
            pickle.dump(mercer_trained_parameters, f)

    else:
        # Mercer (load parameters)
        with open(
            "./mercer_trained_parameters_{}.pkl".format(code),
            "rb",
        ) as f:
            mercer_trained_parameters = pickle.load(f)

        # Favard (load parameters)
        with open(
            "./favard_trained_parameters_{}.pkl".format(code),
            "rb",
        ) as f:
            favard_trained_parameters = pickle.load(f)

        # load the parameters
        mercer_trained_noise = torch.load(
            "./mercer_trained_noise_{}.pt".format(code)
        )
        favard_trained_noise = torch.load(
            "./favard_trained_noise_{}.pt".format(code)
        )

    # Now, we run the comparisons
    """
    Obviously, to be valid we need to take a subset of the inputs and use them
    as input data; and a different subset, and use them as test data.
    """
    test_random_sample_point_count = 50
    input_random_sample_point_count = 600

    input_sample = torch.stack((total_sulphur, free_sulphur), dim=1)
    # print("Shape of input_points:", input_points.shape)
    # print("Shape of output_points:", output_points.shape)
    mercer_gp = get_GP_multidim(
        order,
        mercer_trained_parameters,
        mercer_trained_noise,
        input_sample,
        output_sample,
        GPType.STANDARD,
        KernelType.MERCER,
    )
    favard_gp = get_GP_multidim(
        order,
        favard_trained_parameters,
        favard_trained_noise,
        input_sample,
        output_sample,
        GPType.STANDARD,
        KernelType.FAVARD,
    )
    if not precompared:
        favard_predictive_densities, mercer_predictive_densities = compare_gps(
            favard_gp,
            mercer_gp,
            input_sample,
            output_sample,
            empirical_experiment_count,
            test_random_sample_point_count,
            input_random_sample_point_count,
            # random_sample_point_count,
        )

    fig, ax = plt.subplots()
    density_percentage_increase = (
        torch.Tensor([mercer_predictive_densities])
        - torch.Tensor([favard_predictive_densities])
    ) / torch.Tensor([mercer_predictive_densities])
    positive_counts = len(
        density_percentage_increase[density_percentage_increase > 0]
    )

    breakpoint()
    if dataset == DataSet.RED:
        if not precompared:
            density_diffs = torch.Tensor(
                [favard_predictive_densities]
            ) - torch.Tensor([mercer_predictive_densities])
            torch.save(
                density_diffs,
                "red_density_diffs.pt",
            )
        else:
            density_diffs = torch.load("red_density_diffs.pt")
        plt.hist(density_diffs.numpy().flatten(), bins=150)
        # plt.show()
    elif dataset == DataSet.WHITE:
        if not precompared:
            density_diffs = torch.Tensor(
                [favard_predictive_densities]
            ) - torch.Tensor([mercer_predictive_densities])
            torch.save(
                density_diffs,
                "white_density_diffs.pt",
            )
        else:
            density_diffs = torch.load("white_density_diffs.pt")
        plt.hist(density_diffs.numpy().flatten(), bins=150)
    elif dataset == DataSet.BOTH:
        if not precompared:
            density_diffs = torch.Tensor(
                [favard_predictive_densities]
            ) - torch.Tensor([mercer_predictive_densities])

            torch.save(
                density_diffs,
                "total_density_diffs.pt",
            )
        else:
            density_diffs = torch.load("total_density_diffs.pt")
        plt.hist(density_diffs.numpy().flatten(), bins=150)

    plt.rcParams["text.usetex"] = True
    if save_tikz:
        # plt.rcParams["figure.figsize"] = (6, 4)
        # ax.set_xlabel(r"$\densityfavard - \densitymercer$")
        ax.set_xlabel(
            r"difference in log predictive density: Favard - Mercer$"
        )
        ax.set_ylabel(r"Counts ")
        tikzplotlib.save(
            "/home/william/phd/tex_projects/favard_kernels_aistats2024/diagrams/wine_dataset_multidim{}.tex".format(
                code
            ),
            axis_height="\\winedatasetdiagramheight",
            axis_width="\\winedatasetdiagramwidth",
        )
    else:
        ax.set_xlabel(r"favard - mercer")
        ax.set_ylabel(r"Counts ")
        plt.show()
