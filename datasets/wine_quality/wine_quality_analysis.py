# data found at: https://archive.ics.uci.edu/ml/datasets/Wine+Quality

import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mercergp.builders import (
    build_mercer_gp,
    build_mercer_gp_fourier_posterior,
    train_mercer_params,
    build_smooth_exponential_mercer_gp,
    build_smooth_exponential_mercer_gp_fourier_posterior,
    train_smooth_exponential_mercer_params,
)
from mercergp.MGP import MercerGP, MercerGPFourierPosterior
from mercergp.eigenvalue_gen import SmoothExponentialFasshauer
from mercergp.likelihood import Likelihood
from ortho.builders import get_orthonormal_basis_from_sample
from termcolor import colored

import pickle

torch.set_printoptions(precision=8)


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


def train_basic_mercer_gp(
    non_gaussian_input_sample,
    non_gaussian_output_sample,
    order,
    parameter_loc,
    iter_count=60000,
) -> (MercerGP, MercerGPFourierPosterior, dict):
    """
    This function accepts:
        - an input sample
        - an output sample (of the same size)
        - the order of the model
        - the location of the parameters to load
        - the number of iterations to train for

    It returns:
        - a MercerGP object
        - a MercerGPFourierPosterior object
        - the trained parameters

    """
    # first, set up the initial parameters
    mercer_initial_ard_parameter = torch.Tensor([0.1]).clone()
    mercer_initial_precision_parameter = torch.Tensor(
        [1 / torch.std(non_gaussian_input_sample)]
    ).clone()
    mercer_initial_noise_parameter = torch.Tensor([1.0]).clone()
    mercer_initial_variance_parameter = torch.Tensor([1.0]).clone()
    mercer_initial_ard_parameter.requires_grad = True
    mercer_initial_noise_parameter.requires_grad = True
    mercer_parameters = {
        "noise_parameter": mercer_initial_noise_parameter,
        # "precision_parameter": favard_initial_precision_parameter,
        "precision_parameter": mercer_initial_precision_parameter,
        "ard_parameter": mercer_initial_ard_parameter,
        "degree": 6,  # the eigenvalue polynomial exponent for the decay
        "variance_parameter": mercer_initial_variance_parameter,
    }
    # eigenvalue_generator = FavardEigenvalues(order)
    mercer_optimiser = torch.optim.Adam(
        [mercer_initial_ard_parameter, mercer_initial_noise_parameter],
        lr=0.001,
    )

    # now, either load the parameters or learn them
    print("Training mercer gp!")
    if len(parameter_loc) == 0:
        print("iter_count = ", iter_count)
        trained_mercer_params = train_smooth_exponential_mercer_params(
            mercer_parameters,
            order,
            non_gaussian_input_sample,
            non_gaussian_output_sample,
            mercer_optimiser,
            iter_count=iter_count,
        )
    else:
        trained_mercer_params = torch.load(parameter_loc)

    trained_mercer_params["variance_parameter"] = torch.Tensor([1.0]).clone()
    # with the new parameters, build the normal one or the fourier
    favard_mercer_gp = build_smooth_exponential_mercer_gp(
        trained_mercer_params,
        order,
    )
    favard_mercer_gp_fourier_posterior = (
        build_smooth_exponential_mercer_gp_fourier_posterior(
            trained_mercer_params,
            order,
            dim=1,
            begin=-25,
            end=25,
            frequency=5,
            rff_order=4000,
            prior_variance=4.0,
        )
    )
    favard_mercer_gp.add_data(
        non_gaussian_input_sample, non_gaussian_output_sample
    )
    favard_mercer_gp_fourier_posterior.add_data(
        non_gaussian_input_sample, non_gaussian_output_sample
    )
    return (
        favard_mercer_gp,
        favard_mercer_gp_fourier_posterior,
        trained_mercer_params,
    )


def train_favard_gp(
    non_gaussian_input_sample,
    non_gaussian_output_sample,
    order,
    parameter_loc,
    iter_count=60000,
) -> (MercerGP, MercerGPFourierPosterior, dict):
    """
    This function accepts:
        - an input sample
        - an output sample (of the same size)
        - the order of the model
        - the location of the parameters to load
        - the number of iterations to train for
    It returns:
        - a MercerGP object
        - a MercerGPFourierPosterior object
        - the trained parameters
    """
    favard_initial_ard_parameter = torch.Tensor([0.1]).clone()
    # favard_initial_precision_parameter = clone()
    favard_initial_precision_parameter = torch.Tensor(
        [1 / torch.std(non_gaussian_input_sample)]
    ).clone()
    favard_initial_noise_parameter = torch.Tensor([1.0]).clone()
    favard_initial_variance_parameter = torch.Tensor([1.0]).clone()
    favard_initial_ard_parameter.requires_grad = True
    favard_initial_noise_parameter.requires_grad = True
    favard_parameters = {
        "noise_parameter": favard_initial_noise_parameter,
        # "precision_parameter": favard_initial_precision_parameter,
        "precision_parameter": favard_initial_precision_parameter,
        "ard_parameter": favard_initial_ard_parameter,
        "degree": 6,  # the eigenvalue polynomial exponent for the decay
        "variance_parameter": favard_initial_variance_parameter,
    }

    # eigenvalue_generator = FavardEigenvalues(order)
    favard_optimiser = torch.optim.Adam(
        [favard_initial_ard_parameter, favard_initial_noise_parameter],
        lr=0.001,
    )

    print("Training favard gp!")

    smooth_exponential_eigenvalues = SmoothExponentialFasshauer(order)
    basis = get_orthonormal_basis_from_sample(
        non_gaussian_input_sample, weight_function, order
    )
    if len(parameter_loc) == 0:
        print("iter_count = ", iter_count)
        trained_favard_params = train_mercer_params(
            favard_parameters,
            non_gaussian_input_sample,
            non_gaussian_output_sample,
            basis,
            favard_optimiser,
            smooth_exponential_eigenvalues,
            iter_count=iter_count,
        )
    else:
        trained_favard_params = torch.load(parameter_loc)
    # when we trained, we didn't use a variance parameter. It doesn't matter,
    # we can use a fake one now...
    trained_favard_params["variance_parameter"] = torch.Tensor([1.0]).clone()
    favard_mercer_gp = build_mercer_gp(
        trained_favard_params,
        order,
        basis,
        smooth_exponential_eigenvalues,
    )
    favard_mercer_gp_fourier_posterior = build_mercer_gp_fourier_posterior(
        trained_favard_params,
        order,
        basis,
        smooth_exponential_eigenvalues,
        begin=-25,
        end=25,
        frequency=5,
        rff_order=4000,
        prior_variance=4.0,
    )
    favard_mercer_gp.add_data(
        non_gaussian_input_sample, non_gaussian_output_sample
    )
    favard_mercer_gp_fourier_posterior.add_data(
        non_gaussian_input_sample, non_gaussian_output_sample
    )
    return (
        favard_mercer_gp,
        favard_mercer_gp_fourier_posterior,
        trained_favard_params,
    )


if __name__ == "__main__":
    """
    In the first instance, we will train separately on the white wine and red wine
    data.
    Then, we will conduct a similar approach on the combination of the two,
    which will exhibit multimodality in the input distribution.
    """
    # load data
    red_wine_data = pd.read_csv("winequality-red.csv", sep=";")
    white_wine_data = pd.read_csv("winequality-white.csv", sep=";")
    total_wine_data = pd.concat([red_wine_data, white_wine_data])

    # get some variables to take a look at
    do_hists = False
    do_pairplots = False
    presenting = False

    # build the variables
    # red wine
    rfree_sulphur = torch.Tensor(red_wine_data["free sulfur dioxide"])
    rtotal_sulphur = torch.Tensor(red_wine_data["total sulfur dioxide"])

    # white wine
    wfree_sulphur = torch.Tensor(white_wine_data["free sulfur dioxide"])
    wtotal_sulphur = torch.Tensor(white_wine_data["total sulfur dioxide"])

    # both
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

    if do_hists:  # display the input distributions
        plt.hist(rfree_sulphur.numpy(), bins=50)
        plt.hist(wfree_sulphur.numpy(), bins=157, alpha=0.8)
        plt.hist(free_sulphur.numpy(), bins=203, alpha=0.8, color="green")
        plt.show()
        plt.hist(rtotal_sulphur.numpy(), bins=50)
        plt.hist(wtotal_sulphur.numpy(), bins=157, alpha=0.8)
        plt.hist(total_sulphur.numpy(), bins=203, alpha=0.8, color="purple")
        plt.show()

    if do_pairplots:
        sns.pairplot(red_wine_data, hue="quality")
        sns.pairplot(white_wine_data, hue="quality")
        sns.pairplot(total_wine_data, hue="quality")
        plt.show()

    # build the mercer GP model from the alcohol and density data
    # first, get the hyperparameters

    # convert the data to tensors

    # get descaled x axes for later
    x_axis = torch.linspace(
        -10, 30, 5000
    )  # the frequency of the x axis must be high enough to avoid aliasing
    # in the posterior sampling plot

    #  standardise the data for training
    std_rtotal_sulphur = standardise_data(rtotal_sulphur)
    red_input_mean = torch.mean(rtotal_sulphur)
    red_input_std = torch.std(rtotal_sulphur)

    std_wtotal_sulphur = standardise_data(wtotal_sulphur)
    white_input_mean = torch.mean(wtotal_sulphur)
    white_input_std = torch.std(wtotal_sulphur)

    std_total_sulphur = standardise_data(total_sulphur)
    total_input_mean = torch.mean(total_sulphur)
    total_input_std = torch.std(total_sulphur)

    # noise_parameter = torch.Tensor([10.0])
    # ard_parameter = torch.Tensor([1.0])
    # precision_parameter = torch.Tensor([1.0])
    # noise_parameter.requires_grad = True
    # precision_parameter.requires_grad = False
    # ard_parameter.requires_grad = True
    # parameters = {
    # "noise_parameter": noise_parameter,
    # "ard_parameter": ard_parameter,
    # "precision_parameter": precision_parameter,
    # }

    order = 10
    itercount = 9000
    eigenvalue_generator = SmoothExponentialFasshauer(order)
    get_red_wine_gp = True
    get_white_wine_gp = True
    get_total_wine_gp = True
    get_red_wine_gp_mercer = True
    get_white_wine_gp_mercer = True
    get_total_wine_gp_mercer = True

    """
    Below we present the plots. The first corresponds to the red wine data;
    the second to white wine data.
    """
    if get_red_wine_gp:
        # first, estimate the parameters
        (
            red_wine_gp,
            red_wine_gp_fourier_posterior,
            red_wine_gp_parameters,
        ) = train_favard_gp(
            std_rtotal_sulphur,
            rfree_sulphur,
            order,
            "red_wine_gp_parameters.pt",
            iter_count=itercount,
        )
        torch.save(red_wine_gp_parameters, "red_wine_gp_parameters.pt")

        # plot the red wine observations
        if presenting:
            present_gp(red_wine_gp, x_axis, red_input_mean, red_input_std)
            plt.show()
            present_gp(
                red_wine_gp_fourier_posterior,
                x_axis,
                red_input_mean,
                red_input_std,
            )
            plt.show()

    if get_white_wine_gp:
        # first, estimate the parameters
        (
            white_wine_gp,
            white_wine_gp_fourier_posterior,
            white_wine_gp_parameters,
        ) = train_favard_gp(
            std_wtotal_sulphur,
            wfree_sulphur,
            order,
            "white_wine_gp_parameters.pt",
            iter_count=itercount,
        )
        torch.save(white_wine_gp_parameters, "white_wine_gp_parameters.pt")
        if presenting:
            present_gp(
                white_wine_gp,
                x_axis,
                white_input_mean,
                white_input_std,
            )
            plt.show()
            present_gp(
                white_wine_gp_fourier_posterior,
                x_axis,
                white_input_mean,
                white_input_std,
            )
            plt.show()

    if get_total_wine_gp:
        # first, estimate the parameters
        (
            total_wine_gp,
            total_wine_gp_fourier_posterior,
            total_wine_gp_parameters,
        ) = train_favard_gp(
            std_total_sulphur,
            free_sulphur,
            order,
            "total_wine_gp_parameters.pt",
            iter_count=itercount,
        )
        torch.save(total_wine_gp_parameters, "total_wine_gp_parameters.pt")

        # plot the total wine observations
        if presenting:
            present_gp(
                total_wine_gp, x_axis, total_input_mean, total_input_std
            )
            plt.show()
            present_gp(
                total_wine_gp_fourier_posterior,
                x_axis,
                total_input_mean,
                total_input_std,
            )
            plt.show()

    if get_red_wine_gp_mercer:
        # first, estimate the parameters
        (
            red_wine_gp_mercer,
            red_wine_gp_fourier_posterior_mercer,
            red_wine_gp_parameters_mercer,
        ) = train_basic_mercer_gp(
            std_rtotal_sulphur,
            rfree_sulphur,
            order,
            "red_wine_gp_parameters_mercer.pt",
            iter_count=itercount,
        )
        torch.save(
            red_wine_gp_parameters_mercer, "red_wine_gp_parameters_mercer.pt"
        )

        # plot the red wine observations
        if presenting:
            present_gp(
                red_wine_gp_mercer, x_axis, red_input_mean, red_input_std
            )
            plt.show()
            present_gp(
                red_wine_gp_fourier_posterior_mercer,
                x_axis,
                red_input_mean,
                red_input_std,
            )
            plt.show()

    if get_white_wine_gp_mercer:
        # first, estimate the parameters
        (
            white_wine_gp_mercer,
            white_wine_gp_fourier_posterior_mercer,
            white_wine_gp_parameters_mercer,
        ) = train_basic_mercer_gp(
            std_wtotal_sulphur,
            wfree_sulphur,
            order,
            "white_wine_gp_parameters_mercer.pt",
            iter_count=itercount,
        )
        torch.save(
            white_wine_gp_parameters_mercer,
            "white_wine_gp_parameters_mercer.pt",
        )
        if presenting:
            present_gp(
                white_wine_gp_mercer,
                x_axis,
                white_input_mean,
                white_input_std,
            )
            plt.show()
            present_gp(
                white_wine_gp_fourier_posterior_mercer,
                x_axis,
                white_input_mean,
                white_input_std,
            )
            plt.show()

    if get_total_wine_gp_mercer:
        # first, estimate the parameters
        (
            total_wine_gp_mercer,
            total_wine_gp_fourier_posterior_mercer,
            total_wine_gp_parameters_mercer,
        ) = train_basic_mercer_gp(
            std_total_sulphur,
            free_sulphur,
            order,
            "total_wine_gp_parameters_mercer.pt",
            iter_count=itercount,
        )
        torch.save(
            total_wine_gp_parameters_mercer,
            "total_wine_gp_parameters_mercer.pt",
        )

        # plot the total wine observations
        if presenting:
            present_gp(
                total_wine_gp_mercer, x_axis, total_input_mean, total_input_std
            )
            plt.show()
            present_gp(
                total_wine_gp_fourier_posterior_mercer,
                x_axis,
                total_input_mean,
                total_input_std,
            )
            plt.show()

    # empirics block
    empirical_experiment_count = 1000
    random_sample_point_count = 5
    red_wine_predictive_densities = []
    red_wine_mercer_predictive_densities = []
    white_wine_predictive_densities = []
    white_wine_mercer_predictive_densities = []
    total_wine_predictive_densities = []
    total_wine_mercer_predictive_densities = []

    for i in range(empirical_experiment_count):
        if i % 100 == 0:
            print("Iteration:", i)
        red_wine_random_indices = torch.randint(
            0, len(rtotal_sulphur), (random_sample_point_count,)
        )
        white_wine_random_indices = torch.randint(
            0, len(wtotal_sulphur), (random_sample_point_count,)
        )
        total_wine_random_indices = torch.randint(
            0, len(total_sulphur), (random_sample_point_count,)
        )

        # predictive density values
        if get_red_wine_gp:
            # get the predictive densities
            red_wine_predictive_density = red_wine_gp.get_predictive_density(
                rtotal_sulphur[red_wine_random_indices]
            )
            red_wine_mercer_predictive_density = (
                red_wine_gp_mercer.get_predictive_density(
                    rtotal_sulphur[red_wine_random_indices]
                )
            )
            # get the predictive density values
            red_wine_predictive_density_values = (
                red_wine_predictive_density.log_prob(
                    rfree_sulphur[red_wine_random_indices]
                )
            )
            red_wine_mercer_predictive_density_values = (
                red_wine_mercer_predictive_density.log_prob(
                    rfree_sulphur[red_wine_random_indices]
                )
            )
            print(
                colored("Favard is better", "green")
                if red_wine_predictive_density_values
                > red_wine_mercer_predictive_density_values
                else colored("Mercer is better", "magenta")
            )
            print(
                "favard red wine log predictive density:",
                red_wine_predictive_density_values,
            )
            print(
                "mercer red wine log predictive density:",
                red_wine_mercer_predictive_density_values,
            )
            red_wine_predictive_densities.append(
                red_wine_predictive_density_values
            )
            red_wine_mercer_predictive_densities.append(
                red_wine_mercer_predictive_density_values
            )

            # breakpoint()

        if get_white_wine_gp:
            # get the predictive densities
            white_wine_predictive_density = (
                white_wine_gp.get_predictive_density(
                    wtotal_sulphur[white_wine_random_indices]
                )
            )

            # get the predictive density values
            white_wine_predictive_density_values = (
                white_wine_predictive_density.log_prob(
                    wfree_sulphur[white_wine_random_indices]
                )
            )

            white_wine_mercer_predictive_density = (
                white_wine_gp_mercer.get_predictive_density(
                    wtotal_sulphur[white_wine_random_indices]
                )
            )
            white_wine_mercer_predictive_density_values = (
                white_wine_mercer_predictive_density.log_prob(
                    wfree_sulphur[white_wine_random_indices]
                )
            )
            # breakpoint()

            print(
                colored("Favard is better", "green")
                if white_wine_predictive_density_values
                > white_wine_mercer_predictive_density_values
                else colored("Mercer is better", "magenta")
            )
            print(
                "favard white wine log predictive density:",
                colored(white_wine_predictive_density_values, "yellow"),
            )
            print(
                "mercer white log predictive density:",
                colored(white_wine_mercer_predictive_density_values, "yellow"),
            )

            white_wine_predictive_densities.append(
                white_wine_predictive_density_values
            )
            white_wine_mercer_predictive_densities.append(
                white_wine_mercer_predictive_density_values
            )

        if get_total_wine_gp:
            total_wine_predictive_density = red_wine_gp.get_predictive_density(
                total_sulphur[total_wine_random_indices]
            )
            total_wine_predictive_density_values = (
                total_wine_predictive_density.log_prob(
                    free_sulphur[total_wine_random_indices]
                )
            )
            total_wine_mercer_predictive_density = (
                total_wine_gp_mercer.get_predictive_density(
                    total_sulphur[total_wine_random_indices]
                )
            )
            total_wine_mercer_predictive_density_values = (
                total_wine_mercer_predictive_density.log_prob(
                    free_sulphur[total_wine_random_indices]
                )
            )
            print(
                colored("Favard is better", "green")
                if total_wine_predictive_density_values
                > total_wine_mercer_predictive_density_values
                else colored("Mercer is better", "magenta")
            )
            print(
                "favard total log predictive density:",
                total_wine_predictive_density_values,
            )
            print(
                "mercer total log predictive density:",
                total_wine_mercer_predictive_density_values,
            )
            total_wine_predictive_densities.append(
                total_wine_predictive_density_values
            )
            total_wine_mercer_predictive_densities.append(
                total_wine_mercer_predictive_density_values
            )
        pickle.dump(red_wine_predictive_densities, open("red_wine_pd.p", "wb"))
        pickle.dump(
            red_wine_mercer_predictive_densities,
            open("red_wine_mercer_pd.p", "wb"),
        )
        pickle.dump(
            white_wine_predictive_densities, open("white_wine_pd.p", "wb")
        )
        pickle.dump(
            white_wine_mercer_predictive_densities,
            open("white_wine_mercer_pd.p", "wb"),
        )
        pickle.dump(
            total_wine_predictive_densities, open("total_wine_pd.p", "wb")
        )
        pickle.dump(
            total_wine_mercer_predictive_densities,
            open("total_wine_mercer_pd.p", "wb"),
        )
