import torch
import torch.distributions as D
from mercergp.MGP import MercerGP, MercerKernel, HilbertSpaceElement
from ortho.basis_functions import (
    Basis,
    smooth_exponential_basis_fasshauer,
    smooth_exponential_eigenvalues_fasshauer,
)
from ortho.orthopoly import OrthonormalPolynomial
from ortho.builders import get_orthonormal_basis, get_gammas_from_sample
from mercergp.likelihood import MercerLikelihood

# from mercergp.builders import build_mercer_gp
from builders import build_favard_gp
from mercergp.builders import build_mercer_gp
import matplotlib.pyplot as plt


def target_function(x: torch.Tensor) -> torch.Tensor:
    return 6 * torch.sin(x)


def get_input_sample(sample_size):
    """
    Returns a sample for the purpose of testing different input measures for
    the problem of the regression.
    """
    # sample = D.Normal(0.0, 1).sample([sample_size])
    std = torch.Tensor([1.0])
    mix = D.Categorical(torch.ones(2))
    comp = D.Normal(torch.Tensor([-2, 2]), torch.Tensor([std, std]))
    dist = torch.distributions.MixtureSameFamily(mix, comp)
    sample = dist.sample((sample_size,))
    # plt.hist(sample.numpy().flatten(), bins=20)
    # plt.show()
    return sample


def get_output_sample(input_sample, true_noise_parameter):
    sample_size = len(input_sample)
    output_sample = target_function(input_sample) + D.Normal(
        0.0, true_noise_parameter.squeeze()
    ).sample([sample_size])
    return output_sample


def get_samples(sample_size, true_noise_parameter):
    input_sample = get_input_sample(sample_size)
    return input_sample, get_output_sample(input_sample, true_noise_parameter)


"""
In this script, we test examples where the Favard kernel works with the 
smooth exponential eigenvalues. Given that we are artificially constructing
a kernel as a truncated kernel, the eigenvalues will trivially be correct
because application of the usual Hilbert-Schmidt operator will retrieve the
corresponding eigenvalue.


This specific iteration runs with arbitrary pre-chosen gammas.
"""

use_moment_based_gammas = True
# build the "ground truth"
gp_order = 8
sample_size = 200
fineness = 500
sample_shape = torch.Size([sample_size])
true_noise_parameter = torch.Tensor([0.5])
input_sample = D.Normal(0.0, 2.0).sample(sample_shape)
# noise_sample = D.Normal(0.0, 0.5).sample(sample_shape)
# output_sample = target_function(input_sample) + noise_sample
input_sample, output_sample = get_samples(sample_size, true_noise_parameter)

# true_coefficients = torch.Tensor([1, 2, 4, 7, 6, 2, 4, 1, 1, 1])

if use_moment_based_gammas:
    gammas = get_gammas_from_sample(input_sample, gp_order)
else:
    initial_gamma_val = 6
    gammas = initial_gamma_val * torch.ones(2 * gp_order)
gammas[0] = 1

noise_parameter = torch.Tensor([0.1])
eigenvalue_smoothness_parameter = torch.Tensor([6.0])
eigenvalue_scale_parameter = torch.Tensor([2.0])
shape_parameter = 2.0 * torch.ones(gp_order)

ard_parameter = torch.Tensor([[1.0]])
precision_parameter = torch.Tensor([1.0])
noise_parameter.requires_grad = False
ard_parameter.requires_grad = True
precision_parameter.requires_grad = True
kernel_params = {
    "gammas": gammas,
    "ard_parameter": ard_parameter,
    "precision_parameter": precision_parameter,
    "noise_parameter": noise_parameter,
}

gammas.requires_grad = False
# noise_parameter.requires_grad = True
eigenvalue_smoothness_parameter.requires_grad = False
eigenvalue_scale_parameter.requires_grad = True
shape_parameter.requires_grad = True

"""
Plotting parameters
"""
end_point = 6
x_axis = torch.linspace(-end_point, end_point, fineness)

"""
Build the basis and the Favard GP
"""
orthobasis = get_orthonormal_basis(gp_order, torch.zeros(2 * gp_order), gammas)
# init_fgp = build_favard_gp(
# kernel_params, gp_order, input_sample, output_sample
# )

init_fgp = build_mercer_gp(
    orthobasis,
    ard_parameter,
    precision_parameter,
    noise_parameter,
    gp_order,
    1,
)

"""
Plot some samples before training
"""
init_fgp.add_data(input_sample, output_sample)

sample_gp = init_fgp.gen_gp()
plt.plot(x_axis.detach().numpy(), sample_gp(x_axis).detach().numpy())
plt.scatter(
    input_sample.detach().numpy().flatten(),
    output_sample.detach().numpy().flatten(),
    marker="+",
)
plt.show()

"""
Build the optimiser and likelihood, and train the parameters
"""
optimiser = torch.optim.Adam(
    [param for param in kernel_params.values()], lr=0.01
)
likelihood = MercerLikelihood(
    gp_order,
    optimiser,
    orthobasis,
    input_sample,
    output_sample,
    eigenvalue_generator=lambda params: smooth_exponential_eigenvalues_fasshauer(
        gp_order, params
    ),
)
likelihood.fit(kernel_params)


"""
Having trained the parameters, we can now build the Gaussian process.
"""
# fgp = build_favard_gp(kernel_params, gp_order, input_sample, output_sample)

fgp = build_mercer_gp(
    orthobasis,
    ard_parameter.detach(),
    precision_parameter.detach(),
    noise_parameter.detach(),
    gp_order,
    1,
)
fgp.add_data(input_sample, output_sample)
func_count = 10
plt.scatter(
    input_sample.detach().numpy().flatten(),
    output_sample.detach().numpy().flatten(),
    marker="+",
)
for i in range(func_count):
    sample = fgp.gen_gp()
    plt.plot(x_axis, sample(x_axis))  # 1.3 seems to lead to a good fit??

mean = fgp.get_posterior_mean()
plt.plot(x_axis, mean(x_axis))
plt.show()
