import matplotlib.pyplot as plt
import mercergp.MGP as mgp
import mercergp.likelihood as mgp_likelihood
import mercergp.eigenvalue_gen as eigenvalue_gen
import mercergp.builders as mgp_builders

import ortho.basis_functions as bf
import ortho.builders as ortho_builders
import torch
import torch.distributions as D

"""
This is a synthetic experiment to display the properties of the Favard kernel 
approach. The aim is to present a clear case where the effect of the input 
measure is taken into account. 
It also provides a clear example of how to construct a model with our technique.
"""


def weight_function(x: torch.Tensor) -> torch.Tensor:
    """
    Evaluates the weight function. Usually, e^{-lx^2}
    """
    length = torch.tensor(1.0)
    return torch.exp(-length * x ** 2)


def test_function(x: torch.Tensor) -> torch.Tensor:
    """
    The test function used in an iteration of Daskalakis, Dellaportas and Panos.
    """
    return (1.5 * torch.sin(2 * x) + 0.5 * torch.cos(10 * x) + x / 8).squeeze()


# sample parameters
order = 8
normalise = False
mixing = True
sample_size = 500
sample_shape = torch.Size((sample_size,))
noise_parameter = torch.Tensor([0.5])
noise_sample = D.Normal(0.0, noise_parameter).sample(sample_shape).squeeze()
control_prior_ard_parameter = torch.Tensor([1.0])
control_prior_precision_parameter = torch.Tensor(
    [1.0]
)  # the prior here is correct. Remedy this if relevant

favard_prior_ard_parameter = torch.Tensor([0.5])
favard_prior_precision_parameter = torch.Tensor(
    [0.5]
)  # the prior here is correct. Remedy this if relevant

favard_prior_ard_parameter.requires_grad = True
favard_prior_precision_parameter.requires_grad = True

# control model setup
control_mean = torch.Tensor([0.0])
control_precision = torch.Tensor([1.0])
control_input_measure = D.Normal(control_mean, 1 / control_precision)
control_input_sample = control_input_measure.sample(sample_shape)
control_output_sample = test_function(control_input_sample) + noise_sample
# breakpoint()
control_basis_params = {
    "ard_parameter": control_prior_ard_parameter,
    "precision_parameter": control_prior_precision_parameter,
    "noise_parameter": noise_parameter,
}
control_basis = bf.Basis(
    bf.smooth_exponential_basis_fasshauer, 1, order, control_basis_params
)

# control model prior mercer gaussian process
control_mgp = mgp_builders.build_mercer_gp(
    control_basis,
    control_prior_ard_parameter,
    control_prior_precision_parameter,
    noise_parameter,
    order,
    1,
)


# build the input distribution
mixture_dist = D.Categorical(torch.Tensor([0.2, 0.8]))
means = 1.4 * torch.Tensor([-1.0, 3.0])
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
        (favard_input_measure.sample(sample_shape).squeeze())
        - favard_alpha / favard_beta
    ) / (
        favard_alpha / (favard_beta ** 2)
    )  # * 20

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
favard_basis = ortho_builders.get_orthonormal_basis(
    betas, gammas, order, weight_function
)

# building the kernel for presentation
favard_basis_params = {
    "ard_parameter": favard_prior_ard_parameter,
    "precision_parameter": favard_prior_precision_parameter,
    "noise_parameter": noise_parameter,
}
eigenvalues = bf.smooth_exponential_eigenvalues_fasshauer(order, favard_basis_params)
kernel = mgp.MercerKernel(order, favard_basis, eigenvalues, favard_basis_params)

# favard model prior mercer gaussian process
favard_mgp = mgp_builders.build_mercer_gp(
    favard_basis,
    favard_prior_ard_parameter,
    favard_prior_precision_parameter,
    noise_parameter,
    order,
    1,
)

# plotting x_axis
x_axis = torch.linspace(-6, 6, 1000)

# plot the built kernel
kernel_values = kernel(torch.Tensor([0.0]), x_axis.unsqueeze(1))
# breakpoint()
plt.plot(x_axis, kernel_values.squeeze().detach())
plt.show()

# prior samples
for i in range(5):
    # breakpoint()
    control_gp_sample = control_mgp.gen_gp()
    plt.plot(x_axis, control_gp_sample(x_axis).detach())
    # plt.plot(x_axis, favard_gp_sample(x_axis), ls="dashed")
plt.show()

for i in range(5):
    favard_gp_sample = favard_mgp.gen_gp()
    plt.plot(x_axis, favard_gp_sample(x_axis).detach(), ls="dashed")
plt.show()


control_optimiser = torch.optim.SGD(
    [param for param in control_basis_params.values()], lr=0.001
)
favard_optimiser = torch.optim.SGD(
    [param for param in favard_basis_params.values()], lr=0.001
)
control_eigenvalue_generator = eigenvalue_gen.SmoothExponentialFasshauer(order)
favard_eigenvalue_generator = eigenvalue_gen.SmoothExponentialFasshauer(order)

# training
control_likelihood = mgp_likelihood.MercerLikelihood(
    order,
    control_optimiser,
    control_basis,
    control_input_sample,
    control_output_sample,
    control_eigenvalue_generator,
)
favard_likelihood = mgp_likelihood.MercerLikelihood(
    order,
    favard_optimiser,
    favard_basis,
    control_input_sample,
    control_output_sample,
    favard_eigenvalue_generator,
)


control_prior_ard_parameter.requires_grad = True
control_prior_precision_parameter.requires_grad = True
control_likelihood.fit(control_basis_params)
favard_likelihood.fit(favard_basis_params)


# presentation
