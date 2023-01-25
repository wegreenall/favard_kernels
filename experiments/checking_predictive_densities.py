import torch
import torch.distributions as D
import math
from ortho.basis_functions import (
    smooth_exponential_basis_fasshauer,
    smooth_exponential_eigenvalues_fasshauer,
    Basis,
    CompositeBasis,
    RandomFourierFeatureBasis,
)
from mercergp.MGP import MercerGP
from mercergp.posterior_sampling import MercerSpectralDistribution
from mercergp.kernels import (
    MercerKernel,
    RandomFourierFeaturesKernel,
    SmoothExponentialKernel,
)
import matplotlib.pyplot as plt
from termcolor import colored


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
    # breakpoint()

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


"""
This file will produce diagrams comparing the predictive density for a MercerGP and a standard GP.
"""
a = 1
b = 2
c = 3


def data_func(x):
    return a * x ** 2 + b * x + c


plot_example = True
plot_pure_predictive = True
plot_mercer_predictive = True
test_predictive_densities = True
test_single_predictive_density = False
sample_size = 200

# build a mercer kernel
order = 15  # degree of approximation
dim = 1

# set up the arguments
l_se = torch.Tensor([[1]])
sigma_se = torch.Tensor([1])
sigma_e = torch.Tensor([1])
epsilon = torch.Tensor([1])
kernel_args = {
    "ard_parameter": l_se,
    "variance_parameter": sigma_se,
    "noise_parameter": sigma_e,
    "precision_parameter": epsilon,
}

dist = torch.distributions.Normal(loc=0, scale=epsilon)
inputs = dist.sample([sample_size])
outputs = data_func(inputs) + torch.distributions.Normal(0, 1).sample(inputs.shape)
test_points = torch.linspace(-2, 2, 100)  # .unsqueeze(1)

# Mercer kernel, normal inputs
eigenvalues = smooth_exponential_eigenvalues_fasshauer(order, kernel_args)
basis = Basis(smooth_exponential_basis_fasshauer, 1, order, kernel_args)
test_kernel = MercerKernel(order, basis, eigenvalues, kernel_args)
mercer_gp = MercerGP(basis, order, dim, test_kernel)
mercer_gp.add_data(inputs, outputs)
predictive_density = mercer_gp.get_marginal_predictive_density(test_points)

# Full kernel, normal inputs
se_kernel = SmoothExponentialKernel(kernel_args)
predictive_density_se_kernel = get_predictive_density_se_kernel(
    se_kernel, test_points, inputs, outputs
)


# now check for the predictive density over the outputs
test_sample_size = 5
test_sample_shape = torch.Size([test_sample_size])
test_inputs = D.Normal(0.0, 1.0).sample(test_sample_shape)
test_data = data_func(test_inputs)
testing_predictor_true = get_predictive_density_se_kernel(
    se_kernel, test_inputs, inputs, outputs
)
testing_predictor_mercer = mercer_gp.get_predictive_density(test_inputs)
predictive_marginal_densities_mercer = torch.exp(
    # get_normal_density_log(
    # testing_predictor_mercer.covariance_matrix,
    # testing_predictor_mercer.mean,
    # test_data,
    # )
    testing_predictor_mercer.log_prob(test_data)
)

predictive_marginal_densities_mercer_2 = torch.exp(
    get_normal_density_log(
        testing_predictor_mercer.covariance_matrix,
        testing_predictor_mercer.mean,
        test_data,
    )
    # testing_predictor_mercer.log_prob(test_data)
)
predictive_marginal_densities_true = torch.exp(
    # get_normal_density_log(
    # testing_predictor_true.covariance_matrix,
    # testing_predictor_true.mean,
    # test_data,
    # )
    testing_predictor_true.log_prob(test_data)
)
predictive_marginal_densities_true_2 = torch.exp(
    get_normal_density_log(
        testing_predictor_true.covariance_matrix,
        testing_predictor_true.mean,
        test_data,
    )
    # testing_predictor_true.log_prob(test_data)
)
print(colored(predictive_marginal_densities_mercer, "red"))
print(colored(predictive_marginal_densities_true, "green"))
print(colored(predictive_marginal_densities_mercer_2, "red"))
print(colored(predictive_marginal_densities_true_2, "green"))
# breakpoint()
"""
    Plotting stuff - just the plots, not the rest
    """
# test the inverse
if plot_example:
    test_sample = mercer_gp.gen_gp()  # the problem!
    test_mean = mercer_gp.get_posterior_mean()

    # GP sample
    plt.plot(
        test_points.flatten().numpy(),
        test_sample(test_points).flatten().numpy(),
    )
    # GP mean
    plt.plot(
        test_points.flatten().numpy(),
        test_mean(test_points).flatten().numpy(),
    )
    # true function
    plt.plot(
        test_points.flatten().numpy(),
        data_func(test_points).flatten().numpy(),
        color="yellow",
    )
    # input/output points
    plt.scatter(inputs, outputs, marker="+")

    if plot_pure_predictive:
        # Smooth Exponential predictive density
        plt.plot(
            test_points,
            predictive_density_se_kernel.loc.numpy(),
            color="green",
        )
        plt.plot(
            test_points,
            predictive_density_se_kernel.loc.numpy()
            + torch.diag(predictive_density_se_kernel.covariance_matrix).numpy(),
            color="blue",
        )
        plt.plot(
            test_points,
            predictive_density_se_kernel.loc.numpy()
            - torch.diag(predictive_density_se_kernel.covariance_matrix).numpy(),
            color="blue",
        )
        # Mercer predictive density
        if plot_mercer_predictive:
            plt.plot(
                test_points,
                predictive_density.loc.numpy().flatten(),
                color="purple",
            )
        plt.plot(
            test_points,
            predictive_density.loc.numpy().flatten()
            + predictive_density.scale.numpy().flatten(),
            color="red",
        )
        plt.plot(
            test_points,
            predictive_density.loc.numpy().flatten()
            - predictive_density.scale.numpy().flatten(),
            color="red",
        )
plt.show()
# print("Predictive density:", predictive_density)
