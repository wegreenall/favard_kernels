import matplotlib.pyplot as plt
import mercergp.MGP as mgp
import mercergp.likelihood as mgp_likelihood
import mercergp.builders as mgp_builders

import ortho.basis_functions as bf
import ortho.builders as ortho_builders
from ortho.orthopoly import OrthonormalPolynomial
import torch
import torch.distributions as D

"""
This is a synthetic experiment to display the properties of the Favard kernel 
approach. The aim is to present a clear case where the effect of the input 
measure is taken into account. 
It also provides a clear example of how to construct a model with our technique.
"""
torch.manual_seed(1)


class HardWiredKernel:
    def __init__(
        self,
        order,
        basis: bf.Basis,
        eigenvalues: torch.Tensor,
        kernel_args: dict,
    ):
        self.order = order
        self.basis = basis
        self.eigenvalues = eigenvalues
        self.kernel_args = kernel_args
        return

    def __call__(self, input, test):
        basis_eval_input = self.basis(input)
        basis_eval_test = self.basis(test.squeeze())
        return torch.einsum(
            "ij,j,kj -> ki",
            basis_eval_input,
            self.eigenvalues,
            basis_eval_test,
        )


def weight_function(x: torch.Tensor) -> torch.Tensor:
    """
    Evaluates the weight function. Usually, e^{-lx^2}
    """
    length = torch.tensor(1.0)
    return torch.exp(-length * (x ** 2))


def test_function(x: torch.Tensor) -> torch.Tensor:
    """
    The test function used in an iteration of Daskalakis, Dellaportas and Panos.
    """
    return 1.5 * torch.sin(2 * x) + 0.5 * torch.cos(10 * x) + x / 8


# sample parameters
order = 8
normalise = False
mixing = True
sample_size = 500
sample_shape = torch.Size((sample_size,))
noise_parameter = torch.Tensor([0.5])
noise_sample = D.Normal(0.0, noise_parameter).sample(sample_shape).squeeze()
prior_ard_parameter = torch.Tensor([0.5])
prior_precision_parameter = torch.Tensor(
    [0.5]
)  # the prior here is correct. Remedy this if relevant

# control model setup
control_mean = torch.Tensor([0.0])
control_precision = torch.Tensor([1.0])
control_input_measure = D.Normal(control_mean, 1 / control_precision)
control_input_sample = control_input_measure.sample(sample_shape)
control_output_sample = test_function(control_input_sample) + noise_sample
basis_params = {
    "ard_parameter": prior_ard_parameter,
    "precision_parameter": prior_precision_parameter,
}
smooth_exponential_basis = bf.Basis(
    bf.smooth_exponential_basis_fasshauer, 1, order, basis_params
)

# control model prior mercer gaussian process
control_mgp = mgp_builders.build_mercer_gp(
    smooth_exponential_basis,
    prior_ard_parameter,
    prior_precision_parameter,
    noise_parameter,
    order,
    1,
)

# build the input distributions
mixture_dist = D.Categorical(torch.Tensor([0.3, 0.7]))
means = 1.5 * torch.Tensor([-1.0, 3.0])
variances = torch.Tensor([1.0, 1.0])
component_dist_1 = D.Normal(means, variances)

# favard model setup - gamma input distribution
favard_alpha = torch.Tensor([3.0])
favard_beta = torch.Tensor([3.0])
if mixing:
    favard_input_measure = D.MixtureSameFamily(mixture_dist, component_dist_1)
else:
    favard_input_measure = D.Gamma(favard_alpha, favard_beta)

if not normalise:
    favard_input_sample = favard_input_measure.sample(sample_shape).squeeze()
else:
    favard_input_sample = (
        (
            (favard_input_measure.sample(sample_shape).squeeze())
            - favard_alpha / favard_beta
        )
        / (favard_alpha / (favard_beta ** 2))
        # * 20
    )

favard_output_sample = test_function(favard_input_sample) + noise_sample

# build the favard basis
"""
    To build the orthonormal basis w.r.t the input measure, we calculate its
    exponential moments according to the weight function, and then fix 
    the basis. Then apply this basis to the Mercer Gaussian Process.
"""
moments = ortho_builders.get_moments_from_sample(
    favard_input_sample, 2 * order, weight_function
)
plt.hist(favard_input_sample.numpy().flatten(), bins=60)
plt.show()
betas, gammas = ortho_builders.get_gammas_betas_from_moments(moments, order)
poly = OrthonormalPolynomial(order, betas, gammas)
favard_basis = ortho_builders.get_orthonormal_basis(
    betas, gammas, order, weight_function
)

# building the kernel for presentation
kernel_params = {
    "ard_parameter": prior_ard_parameter,
    "precision_parameter": prior_precision_parameter,
    "noise_parameter": noise_parameter,
}
eigenvalues = bf.smooth_exponential_eigenvalues_fasshauer(order, kernel_params)
kernel = mgp.MercerKernel(order, favard_basis, eigenvalues, kernel_params)
# test_kernel = HardWiredKernel(order, favard_basis, eigenvalues, kernel_params)

# favard model prior mercer gaussian process

# plotting x_axis
x_axis = torch.linspace(-6, 6, 1000)

# plot the built kernel
kernel_values = kernel(torch.Tensor([0.0]), x_axis.unsqueeze(1))
# breakpoint()
plt.plot(x_axis, kernel_values.squeeze())
plt.show()

# plot the basis functions...
# breakpoint()
# for i in range(order):
# plt.plot(x_axis, poly(x_axis, i, dict()))
# plt.show()
plt.plot(x_axis, favard_basis(x_axis))
plt.show()
