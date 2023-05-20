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
from ortho.builders import OrthoBuilder
from termcolor import colored
import pickle
from typing import List, Tuple
import random


class GPType(Enum):
    STANDARD = 1
    FOURIER = 2


class DataSet(Enum):
    RED = 1
    WHITE = 2
    BOTH = 3


class KernelType(Enum):
    MERCER = 1
    FAVARD = 2


def standardise_data(data):
    return (data - torch.mean(data)) / torch.std(data)


def weight_function(input: torch.Tensor):
    return torch.exp(-(input**2) / 4)


def present_gp(
    gp: MercerGP,
    x_axis,
    input_mean,
    input_std,
):
    """
    Presents a GP.

    It takes 5 GP Samples, and plots each of them. Then it scatters
    the datapoints.
    """
    # First, handle the GP samples

    for i in range(5):
        gp_sample = gp.gen_gp()
        plt.plot(
            x_axis * input_std + input_mean,
            gp_sample(x_axis).detach().numpy(),
        )
    """
    The following plots the data and standard posterior samples from the 
    white wine Favard GP with "Fourier features" posterior component.
    """
    # plot the general observations
    plt.scatter(
        (gp.get_inputs() * input_std + input_mean).numpy(),
        gp.get_outputs().numpy(),
        alpha=0.6,
        marker="+",
        color="magenta",
        linewidth=0.5,
    )
    plt.show()
    return


def get_data(type: DataSet, standardise=True):
    """
    Returns the data for the wine quality analysis.
    """
    if type == DataSet.RED:
        # red wine
        red_wine_data = pd.read_csv("winequality-red.csv", sep=";")
        free_sulphur = torch.Tensor(red_wine_data["free sulfur dioxide"])
        total_sulphur = torch.Tensor(red_wine_data["total sulfur dioxide"])

    elif type == DataSet.WHITE:
        white_wine_data = pd.read_csv("winequality-white.csv", sep=";")
        free_sulphur = torch.Tensor(white_wine_data["free sulfur dioxide"])
        total_sulphur = torch.Tensor(white_wine_data["total sulfur dioxide"])

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
        )
        total_sulphur = torch.Tensor(
            pd.concat(
                (
                    red_wine_data["total sulfur dioxide"],
                    white_wine_data["total sulfur dioxide"],
                ),
                ignore_index=True,
            )
        )
    if standardise:
        free_sulphur = standardise_data(free_sulphur)
        total_sulphur = standardise_data(total_sulphur)
    return free_sulphur, total_sulphur


def get_parameters(
    order: int,
    input_sample: torch.Tensor,
    output_sample: torch.Tensor,
    kernel_type: KernelType,
) -> Tuple[dict, torch.Tensor]:
    """
    Returns the trained hyperparameters for the kernel."""
    # initial parameters for the optimisation
    initial_noise = torch.Tensor([1.0])
    initial_parameters = {
        "ard_parameter": torch.Tensor([1.0]),
        "precision_parameter": torch.Tensor([1.0]),
        "noise_parameter": initial_noise,
        "variance_parameter": torch.Tensor([1.0]),
    }
    # now prepare the likelihood

    # kernel
    if kernel_type == KernelType.MERCER:
        basis = bf.Basis(
            bf.smooth_exponential_basis_fasshauer, 1, order, initial_parameters
        )
    elif kernel_type == KernelType.FAVARD:
        # get the basis from the data
        basis = (
            OrthoBuilder(order)
            .set_sample(input_sample)
            .set_weight_function(weight_function)
            .get_orthonormal_basis()
        )

    # eigenvalues
    eigenvalue_generator = SmoothExponentialFasshauer(order)
    eigenvalues = eigenvalue_generator(initial_parameters)

    kernel = MercerKernel(order, basis, eigenvalues, initial_parameters)

    likelihood = Likelihood(
        order,
        kernel,
        input_sample,
        output_sample,
        eigenvalue_generator,
        param_learning_rate=0.00001,
        sigma_learning_rate=0.00001,
    )
    trained_noise, trained_parameters = likelihood.fit(
        initial_noise, initial_parameters, max_iterations=5000
    )
    return trained_parameters, trained_noise


def get_GP(
    order,
    trained_parameters,
    trained_noise,
    input_sample,
    output_sample,
    gp_type,
    kernel_type,
):
    """
    Returns a GP model.
    """
    trained_parameters["noise_parameter"] = trained_noise
    if kernel_type == KernelType.MERCER:
        if gp_type == GPType.STANDARD:
            gp = build_smooth_exponential_mercer_gp_fourier_posterior(
                trained_parameters,
                order,
                dim=1,
                begin=-25,
                end=25,
                frequency=5,
                rff_order=4000,
            )
        elif gp_type == GPType.FOURIER:
            gp = build_smooth_exponential_mercer_gp_fourier_posterior(
                trained_parameters,
                order,
                dim=1,
                begin=-25,
                end=25,
                frequency=5,
                rff_order=4000,
            )

    elif kernel_type == KernelType.FAVARD:
        basis = (
            OrthoBuilder(order)
            .set_sample(input_sample)
            .set_weight_function(weight_function)
            .get_orthonormal_basis()
        )
        smooth_exponential_eigenvalues = SmoothExponentialFasshauer(order)
        # (
        # trained_parameters
        # )
        breakpoint()
        if gp_type == GPType.STANDARD:
            gp = build_mercer_gp(
                trained_parameters,
                order,
                basis,
                smooth_exponential_eigenvalues,
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
    gp.add_data(input_sample, output_sample)
    return gp


def get_randomized_data(
    total_sulphur, free_sulphur, test_input_count, input_count
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
    test_points_indices = sampled_inputs[:test_random_sample_point_count]
    input_points = total_sulphur[input_point_indices]
    output_points = free_sulphur[input_point_indices]

    test_input_points = total_sulphur[test_points_indices]
    test_output_points = free_sulphur[test_points_indices]
    return test_input_points, test_output_points, input_points, output_points


def compare_gps(
    favard_gp,
    mercer_gp,
    total_sulphur,  # the inputs
    free_sulphur,  # the outputs
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
            total_sulphur, free_sulphur, test_input_count, input_count
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


if __name__ == "__main__":
    # get the data
    do_hists = False
    pretrained = True
    empirical_experiment_count = 1000
    # random_sample_point_count = 50
    dataset = DataSet.BOTH
    if dataset == DataSet.RED:
        free_sulphur, total_sulphur = get_data(DataSet.RED, standardise=False)
    elif dataset == DataSet.WHITE:
        free_sulphur, total_sulphur = get_data(
            DataSet.WHITE, standardise=False
        )
    elif dataset == DataSet.BOTH:
        free_sulphur, total_sulphur = get_data(DataSet.BOTH, standardise=False)

    # rfree_sulphur, rtotal_sulphur = get_data(DataSet.RED, standardise=True)
    # wfree_sulphur, wtotal_sulphur = get_data(DataSet.WHITE, standardise=True)
    # if do_hists:  # display the input distributions
    # plt.hist(free_sulphur.numpy(), bins=50)
    # plt.hist(free_sulphur.numpy(), bins=157, alpha=0.8)
    # plt.hist(free_sulphur.numpy(), bins=203, alpha=0.8, color="green")
    # plt.show()
    # plt.hist(total_sulphur.numpy(), bins=50)
    # plt.hist(total_sulphur.numpy(), bins=157, alpha=0.8)
    # plt.hist(otal_sulphur.numpy(), bins=203, alpha=0.8, color="purple")
    # plt.show()

    """
    Train the Gaussian processes.
    """
    if dataset == DataSet.RED:
        code = "red"
    elif dataset == DataSet.WHITE:
        code = "white"
    elif dataset == DataSet.BOTH:
        code = "both"
    # get the parameters
    order = 10
    if not pretrained:
        # calculate the trained parameters
        (
            mercer_trained_parameters,
            mercer_trained_noise,
        ) = get_parameters(
            order, total_sulphur, free_sulphur, KernelType.MERCER
        )
        (
            favard_trained_parameters,
            favard_trained_noise,
        ) = get_parameters(
            order, total_sulphur, free_sulphur, KernelType.FAVARD
        )

        # now save the parameters
        torch.save(
            mercer_trained_noise,
            "analysis_2_data/mercer_trained_noise_{}.pt".format(code),
        )
        torch.save(
            favard_trained_noise,
            "analysis_2_data/favard_trained_noise_{}.pt".format(code),
        )
        with open(
            "analysis_2_data/favard_trained_parameters_{}.pkl".format(code),
            "wb",
        ) as f:
            pickle.dump(favard_trained_parameters, f)
        with open(
            "analysis_2_data/mercer_trained_parameters_{}.pkl".format(code),
            "wb",
        ) as f:
            pickle.dump(mercer_trained_parameters, f)

    else:
        # get the saved parameters
        with open(
            "analysis_2_data/favard_trained_parameters_{}.pkl".format(code),
            "rb",
        ) as f:
            favard_trained_parameters = pickle.load(f)
        with open(
            "analysis_2_data/mercer_trained_parameters_{}.pkl".format(code),
            "rb",
        ) as f:
            mercer_trained_parameters = pickle.load(f)

        # load the parameters
        mercer_trained_noise = torch.load(
            "analysis_2_data/mercer_trained_noise_{}.pt".format(code)
        )
        favard_trained_noise = torch.load(
            "analysis_2_data/favard_trained_noise_{}.pt".format(code)
        )

    """
    Obviously, to be valid we need to take a subset of the inputs and use them
    as input data; and a different subset, and use them as test data.
    """
    test_random_sample_point_count = 50
    input_random_sample_point_count = 600

    # print("Shape of input_points:", input_points.shape)
    # print("Shape of output_points:", output_points.shape)
    mercer_gp = get_GP(
        order,
        mercer_trained_parameters,
        mercer_trained_noise,
        total_sulphur,
        free_sulphur,
        GPType.STANDARD,
        KernelType.MERCER,
    )
    favard_gp = get_GP(
        order,
        favard_trained_parameters,
        favard_trained_noise,
        total_sulphur,
        free_sulphur,
        GPType.STANDARD,
        KernelType.FAVARD,
    )
    favard_predictive_densities, mercer_predictive_densities = compare_gps(
        favard_gp,
        mercer_gp,
        total_sulphur,
        free_sulphur,
        empirical_experiment_count,
        test_random_sample_point_count,
        input_random_sample_point_count,
        # random_sample_point_count,
    )
    if dataset == DataSet.RED:
        pickle.dump(favard_predictive_densities, open("red_wine_pd.p", "wb"))
        pickle.dump(
            mercer_predictive_densities,
            open("red_wine_mercer_pd.p", "wb"),
        )
    elif dataset == DataSet.WHITE:
        pickle.dump(favard_predictive_densities, open("white_wine_pd.p", "wb"))
        pickle.dump(
            mercer_predictive_densities,
            open("white_wine_mercer_pd.p", "wb"),
        )
    elif dataset == DataSet.BOTH:
        pickle.dump(favard_predictive_densities, open("total_wine_pd.p", "wb"))
        pickle.dump(
            mercer_predictive_densities,
            open("total_wine_mercer_pd.p", "wb"),
        )
