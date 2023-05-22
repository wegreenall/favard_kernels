"""
In here we will add the code for the order selection using polynomial 
eigenvalues. First however we will need to implement the EigenvalueGenerator
interface for the PolynomialEigenvalues, to see if it works in the Mercer 
setting.
"""
import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import ortho.basis_functions as bf
from mercergp.likelihood_refit import Likelihood
from mercergp.eigenvalue_gen import PolynomialEigenvalues
from mercergp.MGP import MercerGP, MercerKernel
from mercergp.builders import (
    build_smooth_exponential_mercer_gp,
    build_mercer_gp,
)
from typing import List
import pickle
import tikzplotlib

# now insert a function that does some stuff
if __name__ == "__main__":
    torch.manual_seed(0)
    run_calculations = False
    # 1) get the true function: a mercer GP.
    plot_ground_truth = True
    true_noise_std = 0.5
    true_order = 8
    start_count = 3
    end_count = 25
    parameters = {
        "scale": torch.Tensor([2.0]),
        "shape": torch.Tensor([2.0]),
        "degree": torch.Tensor([2.0]),
        "noise_parameter": torch.Tensor([1.0]),
        "variance_parameter": torch.Tensor([1.0]),
        "precision_parameter": torch.Tensor([0.1]),  # necessary for this basis
        "ard_parameter": torch.Tensor([1.0]),  # necessary for this basis
    }
    true_eigenvalue_generator = PolynomialEigenvalues(true_order)
    true_eigenvalues = true_eigenvalue_generator(parameters)
    true_coeffics = D.Normal(0, true_eigenvalues).sample()

    true_basis = bf.Basis(
        bf.smooth_exponential_basis_fasshauer, 1, true_order, parameters
    )
    true_mercer_gp = build_mercer_gp(
        parameters, true_order, true_basis, true_eigenvalue_generator
    )

    true_function = true_mercer_gp.gen_gp()
    true_coefficients = true_function.get_coefficients()

    # 2) get the data: a set of points sampled from the true function.
    sample_size = 1000
    input_sample = D.Normal(0, 5).sample((sample_size,))
    noise_sample = true_noise_std * D.Normal(0, 1).sample((sample_size,))
    output_sample = true_function(input_sample) + noise_sample

    if plot_ground_truth:
        # plot the true function
        x = torch.linspace(-10, 10, 1000)
        y = true_function(x)

        plt.plot(x, y, label="true function")
        plt.scatter(input_sample, output_sample, label="sample")
        plt.legend()
        plt.show()

    saved_noises: List = []
    saved_eigenvalues: List = []
    saved_completions: List[bool] = []
    saved_scales: List = []
    saved_shapes: List = []
    # saved_degrees: List = []
    if run_calculations:
        # 3) get the likelihood
        for order in range(start_count, end_count):
            initial_noise = torch.Tensor([1.0])
            initial_parameters = {
                "scale": torch.Tensor([1.0]),
                "shape": torch.Tensor([1.0]),
                "degree": torch.Tensor([2.0]),
                "noise_parameter": torch.Tensor([1.0]),
                "variance_parameter": torch.Tensor([1.0]),
                "precision_parameter": torch.Tensor(
                    [0.1]
                ),  # necessary for this basis
            }
            eigenvalue_generator = PolynomialEigenvalues(order)
            eigenvalues = eigenvalue_generator(initial_parameters)
            basis = bf.Basis(
                bf.smooth_exponential_basis_fasshauer, 1, order, parameters
            )
            kernel = MercerKernel(
                order, basis, eigenvalues, initial_parameters
            )
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
            final_eigenvalues = eigenvalue_generator(trained_parameters)
            saved_noises.append(trained_noise)
            saved_scales.append(trained_parameters["scale"])
            saved_shapes.append(trained_parameters["shape"])
            # saved_degrees.append(trained_parameters["degrees"])
            saved_eigenvalues.append(final_eigenvalues)
            saved_completions.append(likelihood.converged)

        print(saved_noises)
        print(saved_eigenvalues)

        # now save the eigenvalues data and the noise data
        with open(
            "polynomial_order_{}_data/eigenvalues.pkl".format(true_order), "wb"
        ) as f:
            pickle.dump(saved_eigenvalues, f)

        with open(
            "polynomial_order_{}_data/noises.pkl".format(true_order), "wb"
        ) as f:
            pickle.dump(saved_noises, f)

        # save the scale parameters
        with open(
            "polynomial_order_{}_data/scales.pkl".format(true_order), "wb"
        ) as f:
            pickle.dump(saved_scales, f)

        # save the shapes parameters
        with open(
            "polynomial_order_{}_data/shapes.pkl".format(true_order), "wb"
        ) as f:
            pickle.dump(saved_shapes, f)

    # i.e. we are going to be loading the calculated values
    with open(
        "polynomial_order_{}_data/eigenvalues.pkl".format(true_order), "rb"
    ) as f:
        saved_eigenvalues = pickle.load(f)

    with open(
        "polynomial_order_{}_data/noises.pkl".format(true_order), "rb"
    ) as f:
        saved_noises = pickle.load(f)

    with open(
        "polynomial_order_{}_data/scales.pkl".format(true_order), "rb"
    ) as f:
        saved_scales = pickle.load(f)

    with open(
        "polynomial_order_{}_data/shapes.pkl".format(true_order), "rb"
    ) as f:
        saved_shapes = pickle.load(f)

    start_plot_count_diff = 0
    end_plot_count_diff = 0
    # now plot the noises
    plt.plot(
        list(
            range(
                start_count + start_plot_count_diff,
                end_count + end_plot_count_diff,
            )
        ),
        saved_noises,
    )
    # plt.plot(list(range(start_count, end_count)), [true_noise_std for _ in range(start_count, end_count)])
    plt.plot(
        list(
            range(
                start_count + start_plot_count_diff,
                end_count + end_plot_count_diff,
            )
        ),
        [
            true_noise_std
            for _ in range(
                start_count + start_plot_count_diff,
                end_count + end_plot_count_diff,
            )
        ],
    )
    # plot the true order as a vertical line
    plt.axvline(x=true_order, color="r", linestyle="--")
    plt.show()
    axis_width = 10
    x_axis = torch.linspace(-axis_width, axis_width, 1000)  # .unsqueeze(1)
    marker_value = "."
    marker_size = 0.5
    # now produce the plot in a tikzplotlib way
    plt.rcParams["text.usetex"] = True
    # plt.rcParams["figure.figsize"] = (6, 4)
    fig, ax = plt.subplots()

    ax.set_xlabel(r"$\mathcal{X}$")
    ax.set_ylabel(r"Function Values")
    # ax.scatter(inputs, outputs, marker=marker_value, s=marker_size)
    ax.plot(
        list(range(start_count, end_count)),
        saved_noises,
        # x_axis,
        # fourier_sample_data.real,
        # x_axis,
        # fourier_sample_data.imag,
        # x_axis,
        # posterior_sample_data,
    )
    plt.axvline(x=true_order, color="r", linestyle="--", linewidth=0.5)
    ax.legend(
        (
            "Ground Truth",
            "Re(Posterior sample)",
            # "Im(Posterior sample)",
            "Posterior mean",
        )
        # fontsize="x-small",
    )

    tikzplotlib.save(
        "/home/william/phd/tex_projects/favard_kernels_neurips/diagrams/order_selection_polynomial_{}.tex".format(
            true_order
        ),
        # "./polynomial_{}_data/noise_plot.tex".format(true_order),
        axis_height="\\orderselectiondiagramheight",
        axis_width="\\orderselectiondiagramwidth",
    )
