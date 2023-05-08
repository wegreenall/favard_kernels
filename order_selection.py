import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import ortho.basis_functions as bf
from mercergp.likelihood_refit import Likelihood
from mercergp.eigenvalue_gen import SmoothExponentialFasshauer
from mercergp.MGP import MercerGP, MercerKernel
from mercergp.builders import (
    build_smooth_exponential_mercer_gp,
)
from typing import List
import pickle

# now insert a function that does some stuff
if __name__ == "__main__":
    torch.manual_seed(0)
    run_calculations = True
    # 1) get the true function: a mercer GP.
    plot_ground_truth = True
    true_noise_std = 0.5
    true_order = 12

    parameters = {
        "ard_parameter": torch.Tensor([0.1]),
        "precision_parameter": torch.Tensor([0.1]),
        "noise_parameter": torch.Tensor([true_noise_std]),
        "variance_parameter": torch.Tensor([1.0]),
    }
    true_eigenvalue_generator = SmoothExponentialFasshauer(true_order)
    true_eigenvalues = true_eigenvalue_generator(parameters)
    true_coeffics = D.Normal(0, true_eigenvalues).sample()

    true_mercer_gp = build_smooth_exponential_mercer_gp(
        parameters,
        true_order,
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
    saved_ards: List = []
    saved_variances: List = []
    if run_calculations:
        # 3) get the likelihood
        for order in range(3, 25):
            initial_noise = torch.Tensor([1.0])
            initial_parameters = {
                "ard_parameter": torch.Tensor([1.0]),
                "precision_parameter": torch.Tensor([1.0]),
                "noise_parameter": initial_noise,
                "variance_parameter": torch.Tensor([1.0]),
            }
            eigenvalue_generator = SmoothExponentialFasshauer(order)
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
            saved_ards.append(trained_parameters["ard_parameter"])
            saved_variances.append(trained_parameters["variance_parameter"])
            saved_eigenvalues.append(final_eigenvalues)
            saved_completions.append(likelihood.converged)

        print(saved_noises)
        print(saved_eigenvalues)

        # now save the eigenvalues data and the noise data
        with open("eigenvalues.pkl", "wb") as f:
            pickle.dump(saved_eigenvalues, f)

        with open("noises.pkl", "wb") as f:
            pickle.dump(saved_noises, f)

        # save the ard parameters
        with open("ards.pkl", "wb") as f:
            pickle.dump(saved_ards, f)

        # save the variance parameters
        with open("variances.pkl", "wb") as f:
            pickle.dump(saved_variances, f)

    else:  # i.e. we are going to be loading the calculated values
        with open("eigenvalues.pkl", "rb") as f:
            saved_eigenvalues = pickle.load(f)

        with open("noises.pkl", "rb") as f:
            saved_noises = pickle.load(f)

        with open("ards.pkl", "rb") as f:
            saved_ards = pickle.load(f)

        with open("variances.pkl", "rb") as f:
            saved_variances = pickle.load(f)

        # now plot the noises
        plt.plot(list(range(3, 18)), saved_noises)
        plt.plot(list(range(3, 18)), [true_noise_std for _ in range(3, 18)])
        plt.show()
