import torch
import torch.distributions as D
import math
import matplotlib.pyplot as plt
import ortho
from ortho.basis_functions import Basis, OrthonormalBasis
from ortho.orthopoly import (
    OrthogonalPolynomial,
    OrthonormalPolynomial,
    OrthogonalBasisFunction,
)
from ortho.builders import (
    get_poly_from_sample,
    get_gammas_from_moments,
    get_moments_from_sample,
    get_gammas_from_sample,
)

from ortho.measure import MaximalEntropyDensity
from mercergp.likelihood import MercerLikelihood
from mercergp.MGP import MercerGP
from mercergp.kernels import MercerKernel
from typing import Dict


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


a = -8
b = 30
c = 4
d = 5


def test_function(x: torch.Tensor) -> torch.Tensor:
    return a * x ** 4 + b * x ** 2 + c * x + d


# plotting parameters
end_point = 5
fineness = 400
# start
coeffic = 6
order = 8
betas = 0 * torch.ones(2 * order)
gammas = coeffic * torch.ones(2 * order)
gammas[0] = 1

noise_parameter = torch.Tensor([2.0])
eigenvalue_smoothness_parameter = torch.Tensor([2.0])
eigenvalue_scale_parameter = torch.Tensor([2.0])
shape_parameter = torch.Tensor([2.0])
noise_parameter.requires_grad = False
eigenvalue_smoothness_parameter.requires_grad = False
eigenvalue_scale_parameter.requires_grad = False
shape_parameter.requires_grad = False

# generate the sample
true_noise_parameter = torch.Tensor([1.0])
sample_size = 200
input_sample = get_input_sample(sample_size)

output_sample = test_function(input_sample) + D.Normal(
    0.0, true_noise_parameter.squeeze()
).sample([sample_size])

# breakpoint()
# gammas = get_gammas_from_sample(input_sample, 2 * order)
# moments = get_moments_from_sample(input_sample, 2 * order)
# gammas_fae_moments = get_gammas_from_moments(moments, order)
# print("true_gammas:", gammas)
# print("calcd_gammas:", gammas_fae_moments)
# breakpoint()
orthopoly = get_poly_from_sample(input_sample, order)

params: Dict[str, torch.Tensor] = dict()

# plotting flags:
plot_orthopoly = True
plot_basis = True
plot_weights = True
plot_orthobasis = True
plot_trained_orthobasis = True
"""
OrthogonalPolynomial
"""
x = torch.linspace(-end_point, end_point, fineness)
for deg in range(order):
    vals = orthopoly(x, deg, params).detach()
    if plot_orthopoly:
        plt.plot(x, vals)
plt.show()

"""
Basis
"""
basis = Basis(orthopoly, 1, order, params)
basis_vals = basis(x).detach()
if plot_basis:
    plt.plot(x, basis_vals)
    plt.show()
"""
weight function:
    Check that the maximal entropy density weight function actually does
    something...
"""
weight_function = MaximalEntropyDensity(order, betas, gammas)
# breakpoint()
weight_vals = weight_function(x).detach()
if plot_weights:
    plt.plot(x, weight_vals)
    plt.show()

"""
OrthonormalBasis:
    Constructs a basis that also includes a weight function...
"""
# weight_function = MaximalEntropyDensity(order, betas, gammas)
orthobasis = OrthonormalBasis(orthopoly, weight_function, 1, order, params)
orthobasis_vals = orthobasis(x).detach()
if plot_orthobasis:
    plt.plot(x, orthobasis_vals)
    plt.show()

# noise_parameter = torch.Tensor([0.2])
# eigenvalue_smoothness_parameter = torch.Tensor([2.0])
# eigenvalue_scale_parameter = torch.Tensor([2.0])
# shape_parameter = torch.Tensor([1.0])
# eigenvalue_smoothness_parameter.requires_grad = False
# eigenvalue_scale_parameter.requires_grad = False
# shape_parameter.requires_grad = False
parameters = {
    "gammas": gammas,
    "noise_parameter": noise_parameter,
    "eigenvalue_smoothness_parameter": eigenvalue_smoothness_parameter,
    "eigenvalue_scale_parameter": eigenvalue_scale_parameter,
    "shape_parameter": shape_parameter,
}
"""
Likelihood:
    Test that the likelihood is behaving reasonably
"""
optimiser = torch.optim.Adam([value for value in parameters.values()], lr=0.01)
# optimiser = torch.optim.SGD([value for value in parameters.values()], lr=0.25)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, threshold=1e-8, factor=0.95
)

# optimiser = torch.optim.RMSprop(
# [value for value in parameters.values()], lr=0.001
# )
likelihood = MercerLikelihood(
    order, optimiser, scheduler, orthobasis, input_sample, output_sample
)

print(parameters)
# likelihood parameters

breakpoint()
print("ABOUT TO FIT THE LIKELIHOOD!")
gammas.requires_grad = True
likelihood.fit(parameters)
final_gammas = gammas.detach()
final_gammas[0] = 1.0
final_betas = torch.zeros(2 * order)
print("ABOUT TO BUILD FINAL ORTHOPOLY!")
final_orthopoly = OrthogonalPolynomial(order, final_betas, final_gammas)

final_orthonormally = OrthonormalPolynomial(
    order, final_betas[: order + 1], final_gammas[: order + 1]
)

final_weight_function = MaximalEntropyDensity(order, final_betas, final_gammas)
trained_basis = OrthonormalBasis(
    final_orthonormally, final_weight_function, 1, order, params
)

trained_orthobasis_vals = trained_basis(x).detach()

print("The orthonormal basis looks like this:")
for deg in range(order):
    # final_orthopoly_vals = final_orthopoly(x, deg, params)
    trained_orthobasis_vals = trained_basis(x).detach()
    # plt.plot(x, final_orthopoly_vals)
    if plot_trained_orthobasis:
        plt.plot(x, trained_orthobasis_vals)
plt.show()

# Now make a GP.
"""
Need to build eigenvalues, and kernel args
"""
# eigenvalues =
params = {
    "ard_parameter": torch.Tensor([0.1]),
    "precision_parameter": torch.Tensor([1.0]),
    "noise_parameter": noise_parameter,
}
eigenvalues = ortho.basis_functions.smooth_exponential_eigenvalues_fasshauer(
    order,
    params,
)
kernel = MercerKernel(order, trained_basis, eigenvalues, params)
mgp = MercerGP(trained_basis, order, 1, kernel)
for i in range(10):
    sample_gp = mgp.gen_gp()
    plt.plot(sample_gp(x))
plt.show()

# inputs = test_function(input_sample) + torch.Normal(0.0, 1.0).sample([])
mgp.add_data(input_sample, output_sample)

plt.scatter(input_sample, output_sample)
for i in range(10):
    sample_gp = mgp.gen_gp()
    plt.plot(x, sample_gp(x))
plt.show()

mean_gp = mgp.get_posterior_mean()
plt.plot(x, mean_gp(x), color="black")
# plt.plot(x, sample_gp(x), color="black")
plt.plot(x, test_function(x), color="red")
plt.show()
