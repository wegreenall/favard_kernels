import torch
import torch.distributions as D
from ortho.orthopoly import OrthogonalPolynomial, OrthogonalBasisFunction
from ortho.basis_functions import Basis
from mercergp.MGP import MercerGP
from mercergp.kernels import MercerKernel
import matplotlib.pyplot as plt


def get_eigenvalues(
    smoothness: int, init_val: torch.Tensor, order: int
) -> torch.Tensor:
    vals = torch.linspace(1, order, order)
    # exponents = smoothness * torch.ones(order)
    return 1 / (vals ** smoothness)


# get the random parameters
order = 10
# betas = D.Normal(0.0, 0.01).sample([2 * order + 1])
betas = 0 * torch.ones(2 * order + 1)

gammas = 1 * torch.ones(2 * order + 1)
# gammas = D.Exponential(8.0).sample([2 * order + 1]) + 7
gammas[0] = 1

# p[otting params
fineness = 1000
test_points = torch.linspace(-2, 2, fineness)


poly = OrthogonalPolynomial(order, betas, gammas)


orthogonal_basis_function = OrthogonalBasisFunction(order, betas, gammas)
basis = Basis(orthogonal_basis_function, 1, order)
params = None
for deg in range(order):
    plt.plot(test_points, orthogonal_basis_function(test_points, deg, params))
plt.show()
# what do kernel_args need to be?
init_val = torch.Tensor([5])
smoothness = 2
eigenvalues = get_eigenvalues(smoothness, init_val, order)
kernel_args = {"noise_parameter": torch.Tensor([0.1])}
kernel = MercerKernel(order, basis, eigenvalues, kernel_args)

start_points = torch.zeros(fineness)
# start_point = torch.Tensor([0.0])
kernel_vals = kernel(start_points, test_points)
# plt.plot(test_points, kernel_vals[0, :].detach().numpy())
plt.show()

my_gp = MercerGP(basis, order, 1, kernel)
# for i in range(10):
# gp_sample = my_gp.gen_gp()
# plt.plot(test_points, gp_sample(test_points).detach().numpy())
plt.show()
