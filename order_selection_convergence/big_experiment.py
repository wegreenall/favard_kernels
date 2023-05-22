import torch
import torch.distributions as D
import matplotlib.pyplot as plt
from ortho.builders import OrthoBuilder
import ortho.basis_functions as bf
from mercergp.builders import (
    build_smooth_exponential_mercer_gp,
    build_mercer_gp,
)
from mercergp.likelihood_refit import Likelihood
from mercergp.eigenvalue_gen import (
    SmoothExponentialFasshauer,
)
from mercergp.MGP import MercerKernel


def argmin(x: list) -> int:
    """
    Returns the index of the minimum element in the list.
    """
    val = min(x)
    return x.index(val)


def weight_function(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-(x**2) / 2)


if __name__ == "__main__":
    true_orders = [8, 12]
    start_count = 4  # starting order
    end_count = 18  # ending order
    orders = range(start_count, end_count)

    # groudn truth hyperparameters
    true_noise_std = 0.5
    true_order = 12
    start_count = 3
    end_count = 25
    experiment_count = 3

    # data hyperparameters
    sample_size = 1000
    # saved_noises = torch.zeros(end_count - start_count, experiment_count)

    for true_order in true_orders:  # i.e. 8 or 12
        parameters = {
            "ard_parameter": torch.Tensor([2.0]),
            # "shape": torch.Tensor([2.0]),
            # "degree": torch.Tensor([4.0]),
            "noise_parameter": torch.Tensor([true_noise_std]),
            "variance_parameter": torch.Tensor([1.0]),
            "precision_parameter": torch.Tensor(
                [0.1]
            ),  # necessary for this basis
        }
        estimated_orders = []
        for _ in range(experiment_count):
            """
            For each experiment, we will get a new set of data, and
            a new set of true coefficients.
            """
            true_eigenvalue_generator = SmoothExponentialFasshauer(true_order)
            true_eigenvalues = true_eigenvalue_generator(parameters)
            input_sample = D.Normal(0, 5).sample((sample_size,))

            true_basis = bf.Basis(
                bf.smooth_exponential_basis_fasshauer,
                1,
                true_order,
                parameters,
            )

            true_mercer_gp = build_mercer_gp(
                parameters,
                true_order,
                true_basis,
                true_eigenvalue_generator,
            )
            true_function = true_mercer_gp.gen_gp()
            true_coefficients = true_function.get_coefficients()

            noise_sample = true_noise_std * D.Normal(0, 1).sample(
                (sample_size,)
            )
            output_sample = true_function(input_sample)

            estimated_noises = []
            for order in orders:  # i.e. try this thing for each of the orders
                """
                For each of the orders, we need to get a likelihood and fit
                the parameters. We need to save the noise parameter, and also
                the true order
                """
                initial_noise = torch.Tensor([1.0])
                initial_parameters = {
                    "ard_parameter": torch.Tensor([1.0]),
                    "noise_parameter": torch.Tensor([1.0]),
                    "variance_parameter": torch.Tensor([1.0]),
                    "precision_parameter": torch.Tensor(
                        [0.1]
                    ),  # necessary for this basis
                }
                eigenvalue_generator = SmoothExponentialFasshauer(order)
                eigenvalues = eigenvalue_generator(initial_parameters)
                basis = (
                    OrthoBuilder(order)
                    .set_sample(input_sample)
                    .set_weight_function(weight_function)
                    .get_orthonormal_basis()
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
                    initial_noise,
                    initial_parameters,
                    max_iterations=5000,
                )
                final_eigenvalues = eigenvalue_generator(trained_parameters)
                estimated_noises.append(trained_noise)
            estimated_orders.append(argmin(estimated_noises) + start_count)

        success_count = 0
        for ord in estimated_orders:
            if ord == true_order:
                success_count += 1
        saved_data = {
            "decay": "polynomial",
            "true_order": true_order,
            "estimated_orders": estimated_orders,
            "success_count": success_count,
        }
