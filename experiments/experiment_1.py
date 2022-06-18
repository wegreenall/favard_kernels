import matplotlib.pyplot as plt
import mercergp.MGP as mgp
import mercergp.likelihood as mgp_likelihood
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
    length = torch.tensor(3.0)
    return torch.exp(-length * x ** 2)


def test_function(x: torch.Tensor) -> torch.Tensor:
    """
    The test function used in an iteration of Daskalakis, Dellaportas and Panos.
    """
    return 1.5 * torch.sin(2 * x) + 0.5 * torch.cos(10 * x) + x / 8


# sample parameters
order = 4
sample_size = 500
sample_shape = torch.Size((sample_size,))
noise_parameter = torch.Tensor([0.5])
noise_sample = D.Normal(0.0, noise_parameter).sample(sample_shape).squeeze()
prior_ard_parameter = torch.Tensor([1.0])
prior_precision_parameter = torch.Tensor(
    [1.0]
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


# favard model setup - gamma input distribution
favard_alpha = torch.Tensor([50.0])
favard_beta = torch.Tensor([0.5])
favard_input_measure = D.Gamma(favard_alpha, favard_beta)
favard_input_sample = (
    (
        (favard_input_measure.sample(sample_shape).squeeze())
        - favard_alpha / favard_beta
    )
    / (favard_alpha / favard_beta ** 2)
) * 20

favard_output_sample = test_function(control_input_sample) + noise_sample

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

# favard model prior mercer gaussian process
favard_mgp = mgp_builders.build_mercer_gp(
    favard_basis,
    prior_ard_parameter,
    prior_precision_parameter,
    noise_parameter,
    order,
    1,
)

breakpoint()

# prior samples
x_axis = torch.linspace(-4, 4, 1000)
for i in range(5):
    # breakpoint()
    control_gp_sample = control_mgp.gen_gp()
    plt.plot(x_axis, control_gp_sample(x_axis))
    # plt.plot(x_axis, favard_gp_sample(x_axis), ls="dashed")
plt.show()

for i in range(5):
    favard_gp_sample = favard_mgp.gen_gp()
    plt.plot(x_axis, favard_gp_sample(x_axis), ls="dashed")
plt.show()


# training

# presentation
