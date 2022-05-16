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
    SymmetricOrthonormalPolynomial,
)
from ortho.builders import (
    get_poly_from_sample,
    get_gammas_from_moments,
    get_moments_from_sample,
    get_gammas_from_sample,
    get_weight_function_from_sample,
)

from favard_kernels.builders import build_favard_gp
import matplotlib.pyplot as plt

a = -8
b = -30
c = -4
d = 5


# def test_function(x: torch.Tensor) -> torch.Tensor:
# return a * x ** 4 + b * x ** 2 + c * x + d

torch.manual_seed(0)


def test_function(x: torch.Tensor) -> torch.Tensor:
    return 10 * torch.sin(x / 4)


def get_input_sample(sample_size):
    """
    Returns a sample for the purpose of testing different input measures for
    the problem of the regression.
    """
    # sample = D.Normal(0.0, 1).sample([sample_size])
    std = torch.Tensor([1.0])
    mix = D.Categorical(torch.ones(2))
    comp = D.Normal(torch.Tensor([-0, 0]), torch.Tensor([std, std]))
    dist = torch.distributions.MixtureSameFamily(mix, comp)
    sample = dist.sample((sample_size,))
    # plt.hist(sample.numpy().flatten(), bins=20)
    # plt.show()
    return sample


def get_output_sample(input_sample, true_noise_parameter):
    sample_size = len(input_sample)
    output_sample = test_function(input_sample) + D.Normal(
        0.0, true_noise_parameter.squeeze()
    ).sample([sample_size])
    return output_sample


def get_samples(sample_size, true_noise_parameter):
    input_sample = get_input_sample(sample_size)
    return input_sample, get_output_sample(input_sample, true_noise_parameter)


# final_orthonormally = OrthonormalPolynomial(order, torch.zeros(order), gammas)
dim = 1
order = 8
noise_parameter = torch.Tensor([0.5])
eigenvalue_smoothness_parameter = torch.Tensor([2.0])
eigenvalue_scale_parameter = torch.Tensor([2.0])
shape_parameter = 2.0 * torch.ones(order)

"""
Set up the gammas
"""
coeffic = 1
gammas = coeffic * torch.ones(2 * order)
gammas[0] = 1

sample_size = 100
true_noise_parameter = torch.Tensor([0.5])
input_sample, output_sample = get_samples(sample_size, true_noise_parameter)

parameters = {
    "gammas": gammas,
    "noise_parameter": noise_parameter,
    "eigenvalue_smoothness_parameter": eigenvalue_smoothness_parameter,
    "eigenvalue_scale_parameter": eigenvalue_scale_parameter,
    "shape_parameter": shape_parameter,
}
optimiser = torch.optim.Adam(
    [value for value in parameters.values()], lr=0.001
)
# optimiser = torch.optim.SGD([value for value in parameters.values()], lr=0.005)
# optimiser = torch.optim.SGD([value for value in parameters.values()], lr=0.25)
orthopoly = SymmetricOrthonormalPolynomial(order, gammas)
weight_function = MaximalEntropyDensity(order, torch.zeros(2 * order), gammas)
orthobasis = OrthonormalBasis(orthopoly, weight_function, 1, order, None)

gammas.requires_grad = True
noise_parameter.requires_grad = False
eigenvalue_smoothness_parameter.requires_grad = False
eigenvalue_scale_parameter.requires_grad = False
shape_parameter.requires_grad = False

likelihood = MercerLikelihood(
    order, optimiser, orthobasis, input_sample, output_sample
)

# plotting
fineness = 400
x = torch.linspace(-4, 4, fineness)
plt.plot(x, test_function(x))
plt.scatter(input_sample, output_sample)
plt.show()
likelihood.fit(parameters)

"""
Now build the GP with these parameters
"""
mgp = build_favard_gp(parameters, order)
mgp.add_data(input_sample, output_sample)

plt.scatter(input_sample, output_sample)
for i in range(10):
    sample_gp = mgp.gen_gp()
    plt.plot(x, sample_gp(x))
plt.show()

mean_gp = mgp.get_posterior_mean()
plt.plot(x, mean_gp(x), color="black")
plt.plot(x, test_function(x), color="red")
plt.show()
