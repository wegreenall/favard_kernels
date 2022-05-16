import torch
import torch.distributions as D
from mercergp.builders import build_mercer_gp
from mercergp.likelihood import MercerLikelihood
import matplotlib.pyplot as plt
from ortho.basis_functions import OrthonormalBasis
from ortho.measure import MaximalEntropyDensity
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
    get_weight_function_from_sample,
)

a = -8
b = 30
c = 4
d = 5


def data_func(x):
    return a * x ** 4 + b * x ** 2 + c * x + d


dim = 1
order = 8
ard_parameter = torch.Tensor([1.0])
precision_parameter = torch.Tensor([1.0])
noise_parameter = torch.Tensor([1.0])
mercer_gp = build_mercer_gp(
    ard_parameter, precision_parameter, noise_parameter, order, dim
)

sample_size = 100

x = torch.linspace(-2, 2, 100)
epsilon = torch.Tensor([1.0])
dist = torch.distributions.Normal(loc=0, scale=epsilon)
epsilon = torch.Tensor([1])
inputs = dist.sample([sample_size]).squeeze()
# breakpoint()
data_points = data_func(inputs) + torch.distributions.Normal(0, 1).sample(
    inputs.shape
)

mercer_gp.add_data(inputs, data_points)
mercer_gp_mean = mercer_gp.get_posterior_mean()
plt.plot(x, mercer_gp_mean(x))
plt.scatter(inputs, data_points)
plt.plot(x, data_func(x))
plt.show()

# breakpoint()
# final_orthonormally = OrthonormalPolynomial(order, torch.zeros(order), gammas)
noise_parameter = torch.Tensor([0.1])
eigenvalue_smoothness_parameter = torch.Tensor([2.0])
eigenvalue_scale_parameter = torch.Tensor([2.0])
shape_parameter = 2.0 * torch.ones(order)
noise_parameter.requires_grad = False
eigenvalue_smoothness_parameter.requires_grad = True
eigenvalue_scale_parameter.requires_grad = True
shape_parameter.requires_grad = True
weight_function = get_weight_function_from_sample(inputs, order)
gammas = get_gammas_from_sample(inputs, order)
orthopoly = get_poly_from_sample(inputs, order)
orthobasis = OrthonormalBasis(orthopoly, weight_function, 1, order)
parameters = {
    "noise_parameter": noise_parameter,
    "eigenvalue_smoothness_parameter": eigenvalue_smoothness_parameter,
    "eigenvalue_scale_parameter": eigenvalue_scale_parameter,
    "shape_parameter": shape_parameter,
}
optimiser = torch.optim.Adam([value for value in parameters.values()], lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, threshold=1e-8, factor=0.95
)
likelihood = MercerLikelihood(
    order, optimiser, scheduler, orthobasis, inputs, data_points
)

likelihood.fit(parameters)

print(parameters)
