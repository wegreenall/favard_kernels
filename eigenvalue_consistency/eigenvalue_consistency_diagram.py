import torch
import torch.distributions as D
import matplotlib.pyplot as plt
from termcolor import colored

import math
from mercergp.eigenvalue_gen import SmoothExponentialFasshauer
from ortho.basis_functions import Basis, smooth_exponential_basis_fasshauer
from mercergp.kernels import MercerKernel
import tikzplotlib

run_calculations = True
relative = False
save_tikz = False

order = 11
eigenvalue_generator = SmoothExponentialFasshauer(order)

kernel_args = {
    "precision_parameter": torch.Tensor([2.0]),
    "noise_parameter": torch.Tensor([0.1]),
    "ard_parameter": torch.Tensor([[1.0]]),
    "variance_parameter": torch.Tensor([1.0]),
}
eigenvalues = eigenvalue_generator(kernel_args)
exponents = torch.linspace(0, order - 1, order)
denom = 3 + math.sqrt(8)
true_eigenvalues = math.sqrt(4 / denom) * torch.pow(1 / denom, exponents)
result = torch.allclose(eigenvalues, true_eigenvalues)
print(colored("#########", "blue"))
print(colored(result, "green" if result else "red"))
print(colored("#########", "blue"))

# sample_sizes = (50, 1000, 4000)
sample_sizes = (50, 1000)
gaussian_input_dist = D.Normal(0, 0.5)
categorical_dist = D.Categorical(torch.Tensor([0.2, 0.8]))
component_dist = D.Normal(torch.Tensor([-4, 4]), torch.Tensor([1, 1]))
mixture_dist = D.MixtureSameFamily(categorical_dist, component_dist)

# how many repetitions to calculate it?
repetition_count = 1000

basis = Basis(smooth_exponential_basis_fasshauer, 1, order, kernel_args)
mercer_kernel = MercerKernel(order, basis, eigenvalues, kernel_args)
results = torch.zeros(repetition_count, order, len(sample_sizes), 2)
if run_calculations:
    for rep in range(repetition_count):
        for j, size in enumerate(sample_sizes):
            print(
                "Experiment rep and sample size: {} in progress!".format(
                    (str(rep), str(size))
                )
            )
            sample_shape = torch.Size([size])
            gaussian_input_sample = gaussian_input_dist.sample(sample_shape)
            non_gaussian_input_sample = mixture_dist.sample(sample_shape)

            gaussian_input_kernel_matrix = mercer_kernel(
                gaussian_input_sample, gaussian_input_sample
            )
            non_gaussian_input_kernel_matrix = mercer_kernel(
                non_gaussian_input_sample, non_gaussian_input_sample
            )
            gaussian_input_eigenvalues_estimate = (
                torch.linalg.eigvals(gaussian_input_kernel_matrix) / size
            )[:order]
            non_gaussian_input_eigenvalues_estimate = (
                torch.linalg.eigvals(non_gaussian_input_kernel_matrix) / size
            )[:order]
            # print("Gaussian input eigenvalue estimate:")
            # print(colored(gaussian_input_eigenvalues_estimate, "blue"))
            # print("Non-Gaussian input eigenvalue estimate:")
            # print(colored(non_gaussian_input_eigenvalues_estimate, "green"))
            # breakpoint()
            if relative:
                results[rep, :, j, 0] = (
                    torch.abs(
                        eigenvalues - gaussian_input_eigenvalues_estimate.real
                    )
                    / eigenvalues
                )
                results[rep, :, j, 1] = (
                    torch.abs(
                        eigenvalues
                        - non_gaussian_input_eigenvalues_estimate.real
                    )
                    / eigenvalues
                )
            else:
                results[rep, :, j, 0] = torch.abs(
                    eigenvalues - gaussian_input_eigenvalues_estimate.real
                )
                results[rep, :, j, 1] = torch.abs(
                    eigenvalues - non_gaussian_input_eigenvalues_estimate.real
                )

    if relative:
        torch.save(results, "./eigenvalues_samples_relative.pt")
    else:
        torch.save(results, "./eigenvalues_samples.pt")


# gaussian eigenvalues
if relative:
    loaded_results = torch.load("./eigenvalues_samples_relative.pt")
else:
    loaded_results = torch.load("./eigenvalues_samples.pt")
for j, size in enumerate(sample_sizes):
    gaussian_std = torch.std(loaded_results[:, :, j, 0], dim=0)
    gaussian_lower_quantile = torch.quantile(
        loaded_results[:, :, j, 0], 0.25, dim=0
    )
    gaussian_upper_quantile = torch.quantile(
        loaded_results[:, :, j, 0], 0.75, dim=0
    )
    non_gaussian_std = torch.std(loaded_results[:, :, j, 1], dim=0)
    non_gaussian_lower_quantile = torch.quantile(
        loaded_results[:, :, j, 1], 0.25, dim=0
    )
    non_gaussian_upper_quantile = torch.quantile(
        loaded_results[:, :, j, 1], 0.75, dim=0
    )
    gaussian_mean = torch.mean(loaded_results[:, :, j, 0], dim=0)
    non_gaussian_mean = torch.mean(loaded_results[:, :, j, 1], dim=0)

    plt.rcParams["text.usetex"] = True
    plt.rcParams["figure.figsize"] = (6, 4)
    fig, ax = plt.subplots()
    x = torch.linspace(0, order - 1, order)
    ax.set_xlabel(r"$i$")

    ax.plot(
        x,
        gaussian_mean,
        x,
        non_gaussian_mean,
    )
    if j == 0:
        ax.legend(("Gaussian Inputs", "Non-Gaussian Inputs"))
    ax.plot(
        x,
        gaussian_upper_quantile,
        x,
        gaussian_lower_quantile,
        x,
        non_gaussian_upper_quantile,
        x,
        non_gaussian_lower_quantile,
        linestyle="--",
    )
    if save_tikz:
        tikzplotlib.save(
            "/eigenvalueconsistencydiagram" + str(j) + ".tex",
            axis_height="\\eigenvalueconsistencydiagramheight",
            axis_width="\\eigenvalueconsistencydiagramwidth",
        )
    else:
        plt.show()
    # plt.savefig(
    # "/home/william/phd/tex_projects/favard_kernels_icml/diagrams/eigenvalueconsistencydiagram"
    # + str(j)
    # + ".eps",
    # format="eps",
    # dpi=1200,
    # )
# plt.plot(gaussian_mean, color="red")
# plt.plot(non_gaussian_mean, color="green")
# plt.plot(gaussian_upper_quantile, color="blue", linestyle="dashdot")
# plt.plot(gaussian_lower_quantile, color="blue", linestyle="dashdot")
# plt.plot(non_gaussian_upper_quantile, color="magenta", linestyle="dashdot")
# plt.plot(non_gaussian_lower_quantile, color="magenta", linestyle="dashdot")
# plt.plot(gaussian_mean + 2 * gaussian_std, color="blue", linestyle="dashdot")
# plt.plot(gaussian_mean - 2 * gaussian_std, color="blue", linestyle="dashdot")
# plt.plot(
# non_gaussian_mean + 2 * non_gaussian_std, color="magenta", linestyle="dashdot"
# )
# plt.plot(
# non_gaussian_mean - 2 * non_gaussian_std, color="magenta", linestyle="dashdot"
# )
# plt.show()

# print(gaussian_std.shape)

# print(non_gaussian_std.shape)
