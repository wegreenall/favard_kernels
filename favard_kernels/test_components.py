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

from ortho.measure import MaximalEntropyDensity
from mercergp.likelihood import MercerLikelihood
from mercergp.MGP import MercerGP
from mercergp.kernels import MercerKernel

# torch.manual_seed(3)


def test_function(x: torch.Tensor):
    """
    A function to learn from a regression using Gaussian Proculoids
    """
    return 2 * torch.exp(-(x ** 2)) * (torch.sin(x) ** 2)


# plotting parameters
end_point = 4
fineness = 200
# start
coeffic = 10
order = 8
betas = 0 * torch.ones(2 * order + 1)
gammas = coeffic * torch.ones(2 * order + 1)
# gammas[0] = 1
gammas.requires_grad = True

orthopoly = OrthogonalPolynomial(order, betas, gammas)
orthonormalpoly = OrthonormalPolynomial(order, betas, gammas)
params = None

"""
OrthogonalPolynomial
"""
x = torch.linspace(-end_point, end_point, fineness)
for deg in range(order):
    vals = orthopoly(x, deg, params).detach()
    # plt.plot(x, vals)
# plt.show()

"""
Basis
"""
basis = Basis(orthopoly, 1, order, params)
basis_vals = basis(x).detach()
# plt.plot(x, basis_vals)
# plt.show()
"""
weight function:
    Check that the maximal entropy density weight function actually does
    something...
"""
weight_function = MaximalEntropyDensity(order, betas, gammas)
# orthobasis = OrthonormalBasis(orthopoly, weight_function, 1, order, params)
weight_vals = weight_function(x).detach()
# plt.plot(x, weight_vals)
# plt.show()

"""
OrthonormalBasis:
    Constructs a basis that also includes a weight function...
"""
# weight_function = MaximalEntropyDensity(order, betas, gammas)
orthobasis = OrthonormalBasis(
    orthonormalpoly, weight_function, 1, order, params
)
orthobasis_vals = orthobasis(x).detach()
# plt.plot(x, orthobasis_vals)
# plt.show()

"""
Likelihood:
    Test that the likelihood is behaving reasonably
"""
optimiser = torch.optim.Adam([gammas], lr=0.5)
sample_size = 200
input_sample = D.Normal(0.0, 4.0).sample([sample_size])
output_sample = test_function(input_sample) + D.Normal(0.0, 0.01).sample(
    [sample_size]
)
likelihood = MercerLikelihood(
    order, optimiser, orthobasis, input_sample, output_sample
)

# likelihood parameters
noise_parameter = torch.Tensor([0.01])
parameters = {
    "gammas": gammas,
    "noise_parameter": noise_parameter,
    "eigenvalue_smoothness_parameter": torch.Tensor([4.0]),
    "eigenvalue_scale_parameter": torch.Tensor([4.0]),
    "shape_parameter": torch.Tensor([1.0]),
}

likelihood.fit(parameters)
final_gammas = gammas.detach()
final_gammas[0] = 1.0
final_betas = torch.zeros(2 * order + 1)
# breakpoint()
final_orthopoly = OrthogonalPolynomial(
    order, final_betas[:order], final_gammas[:order]
)

final_orthonormally = OrthonormalPolynomial(
    order, final_betas[:order], final_gammas[:order]
)
# breakpoint()
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
    # plt.plot(sample_gp(x))
# plt.show()

# inputs = test_function(input_sample) + torch.Normal(0.0, 1.0).sample([])
mgp.add_data(input_sample, output_sample)

plt.scatter(input_sample, output_sample)
for i in range(10):
    sample_gp = mgp.gen_gp()
    plt.plot(x, sample_gp(x))
plt.show()
