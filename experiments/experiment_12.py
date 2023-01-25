import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import math

from ortho.basis_functions import (
    smooth_exponential_basis_fasshauer,
    smooth_exponential_eigenvalues_fasshauer,
    Basis,
    CompositeBasis,
    RandomFourierFeatureBasis,
)
from ortho.builders import get_orthonormal_basis_from_sample

from mercergp.builders import (
    train_smooth_exponential_mercer_params,
    build_smooth_exponential_mercer_gp,
    train_mercer_params,
    build_mercer_gp,
)

from mercergp.MGP import MercerGP

# from mercergp.posterior_sampling import NonStationarySpectralDistribution
from mercergp.kernels import (
    MercerKernel,
    RandomFourierFeaturesKernel,
    SmoothExponentialKernel,
)
from mercergp.eigenvalue_gen import SmoothExponentialFasshauer
import matplotlib.pyplot as plt
from termcolor import colored
from experiment_functions import test_function, weight_function

"""
This experiment generates the predictive density of:
    - the true GP with full kernel
    - the Mercer GP, 
    - the Favard GP, 
under the inputs:
    - Normal(0, 1)
    - A sequence of ever more "distant" mixture input distributions.
and compares the KL divergences
"""


def mean_generator(i):
    return 1 * i


def get_normal_density_log(
    covariance_matrix: torch.Tensor,
    mean_vector: torch.Tensor,
    value_vector: torch.Tensor,
) -> torch.Tensor:
    k = covariance_matrix.shape[0]
    term_1 = -k / 2 * math.log(2 * math.pi)
    term_2 = -0.5 * torch.logdet(covariance_matrix)
    print("determinant:", torch.exp(-term_2))
    inverse_matrix = torch.inverse(covariance_matrix)
    term_3 = -0.5 * (
        (value_vector - mean_vector).t() @ inverse_matrix @ (value_vector - mean_vector)
    )
    return term_1 + term_2 + term_3


def get_posterior_mean(
    kernel: SmoothExponentialKernel,
    test_points: torch.Tensor,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
):
    noise_parameter = kernel.get_params()["noise_parameter"]
    kernel_matrix_inv = kernel.kernel_inverse(inputs)
    kernel_left = kernel(inputs, test_points)
    result = kernel_left @ kernel_matrix_inv @ outputs
    return result.squeeze()


def get_predictive_density_se_kernel(
    kernel: SmoothExponentialKernel,
    test_points: torch.Tensor,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
) -> D.Distribution:
    # calculate the mean
    posterior_predictive_mean = get_posterior_mean(kernel, test_points, inputs, outputs)

    # now calculate the variance
    posterior_predictive_variance = (
        kernel(test_points, test_points)
        - kernel(inputs, test_points)
        @ kernel.kernel_inverse(inputs)
        @ kernel(test_points, inputs)
        # + kernel.kernel_args["noise_parameter"] ** 2
    )

    # add jitter for positive definiteness
    posterior_predictive_variance += 0.00001 * torch.eye(len(test_points))
    try:
        distribution = D.MultivariateNormal(
            posterior_predictive_mean,
            posterior_predictive_variance,
        )
    except ValueError:
        distribution = D.MultivariateNormal(
            posterior_predictive_mean,
            posterior_predictive_variance,
        )
        print("FOUND NEGATIVE VARIANCE!")
    return distribution


if __name__ == "__main__":
    """
    The program begins here
    """
    normal_test_inputs = True
    # metaparameters
    experiment_count = 1000
    mixtures_count = 6

    # set up the arguments
    order = 15  # degree of approximation
    dim = 1
    sample_size = 4000

    l_se = torch.Tensor([[1]])
    sigma_se = torch.Tensor([1])
    sigma_e = torch.Tensor([0.2])
    epsilon = torch.Tensor([1])
    kernel_args = {
        "ard_parameter": l_se,
        "variance_parameter": sigma_se,
        "noise_parameter": sigma_e,
        "precision_parameter": epsilon,
    }
    variance = (
        0.8  # the variance of each of the components of the experimental mixtures.
    )
    test_inputs_sample_size = 15
    test_input_sample_shape = torch.Size([test_inputs_sample_size])
    if normal_test_inputs:
        test_inputs_distribution = D.Normal(0.0, 1.0)
    else:
        mixing_distribution = D.Categorical(torch.Tensor([0.2, 0.8]))
        component_distribution = D.Normal(
            torch.Tensor([-mean_generator(2), mean_generator(2)]),
            torch.Tensor([variance, variance]),
        )
        test_inputs_distribution = D.MixtureSameFamily(
            mixing_distribution, component_distribution
        )

    se_kernel = SmoothExponentialKernel(kernel_args)
    kls = torch.zeros(experiment_count, mixtures_count, 2)
    for i in range(experiment_count):
        for j in range(mixtures_count):
            print("Experiment {} in progress!".format((str(i), str(j))))
            test_inputs_sample = test_inputs_distribution.sample(
                test_input_sample_shape
            )

            # make the inputs
            mean = mean_generator(j)

            # input data
            mixing_distribution = D.Categorical(torch.Tensor([0.2, 0.8]))
            component_distribution = D.Normal(
                torch.Tensor([-mean, mean]), torch.Tensor([variance, variance])
            )
            dist = D.MixtureSameFamily(mixing_distribution, component_distribution)

            inputs = dist.sample([sample_size])
            outputs = test_function(inputs) + torch.distributions.Normal(0, 1).sample(
                inputs.shape
            )

            # Full kernel
            true_predictive_density = get_predictive_density_se_kernel(
                se_kernel, test_inputs_sample, inputs, outputs
            )

            # shared parameters
            eigenvalues = smooth_exponential_eigenvalues_fasshauer(order, kernel_args)
            eigenvalue_generator = SmoothExponentialFasshauer(order)

            # Mercer kernel
            mercer_basis = Basis(
                smooth_exponential_basis_fasshauer, 1, order, kernel_args
            )
            # test_kernel = MercerKernel(order, mercer_basis, eigenvalues, kernel_args)
            # mercer_gp = MercerGP(mercer_basis, order, dim, test_kernel)
            mercer_gp = build_mercer_gp(
                kernel_args, order, mercer_basis, eigenvalue_generator
            )
            mercer_gp.add_data(inputs, outputs)
            try:
                mercer_predictive_density = mercer_gp.get_predictive_density(
                    test_inputs_sample
                )
            except ValueError:
                print(
                    "Failed to acquire Mercer predictive density for experiment {} ".format(
                        (str(i), str(j))
                    )
                )
                continue

            # Favard kernel
            favard_basis = get_orthonormal_basis_from_sample(
                inputs, weight_function, order
            )
            favard_gp = build_mercer_gp(
                kernel_args, order, favard_basis, eigenvalue_generator
            )
            favard_gp.add_data(inputs, outputs)
            try:
                favard_predictive_density = favard_gp.get_predictive_density(
                    test_inputs_sample
                )
            except ValueError:
                print(
                    "Failed to acquire Favard predictive density for experiment {} ".format(
                        (str(i), str(j))
                    )
                )
                continue

            # print(colored(true_predictive_density.mean, "blue"))
            # print(colored(mercer_predictive_density.mean, "red"))
            # print(colored(favard_predictive_density.mean, "green"))

            # now calculate the predictive densities
            mercer_kl = D.kl.kl_divergence(
                mercer_predictive_density, true_predictive_density
            )
            favard_kl = D.kl.kl_divergence(
                favard_predictive_density, true_predictive_density
            )
            print(colored(mercer_kl, "yellow"))
            print(colored(favard_kl, "magenta"))
            kls[i, j, 0] = mercer_kl
            kls[i, j, 1] = favard_kl
    if normal_test_inputs:
        torch.save(kls, "./experiment_12_data/kl_divergences_normal_test_inputs.pt")
    else:
        torch.save(kls, "./experiment_12_data/kl_divergences_mixture_test_inputs.pt")
