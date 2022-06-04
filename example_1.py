import torch
import torch.distributions as D
from builders import build_favard_gp
from ortho.builders import get_symmetric_orthonormal_basis
from mercergp.likelihood import FavardLikelihood as MercerLikelihood

from ortho.basis_functions import OrthonormalBasis
import matplotlib.pyplot as plt

# from ortho.orthopoly import

"""
This will contain an example of the simple case:
    Basis orthogonal w.r.t arbitrary measure.

To build an example, the following parameters are required:

    - input_sample, a torch.Tensor containing the inputs corresponding to
                    the targets.
    - output_sample: a torch.Tensor containing the targets
    - basis: ortho.basis_functions.OrthonormalBasis
    - optimiser: torch.optim.Optimizer
    - order: int
    - parameters: a Python Dict() containing a values for the following keys:
        - gammas
        - noise_parameter
        - eigenvalue_smoothness_parameter
        - eigenvalue_scale_parameter
        - shape_parameter
"""


def test_function(x: torch.Tensor) -> torch.Tensor:
    return 10 * torch.sin(x / 2)


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


order = 8
true_noise_parameter = torch.Tensor([0.5])
sample_size = 200
input_sample, output_sample = get_samples(sample_size, true_noise_parameter)

# build the initial parameters:
dim = 1
noise_parameter = torch.Tensor([0.5])
eigenvalue_smoothness_parameter = torch.Tensor([6.0])
eigenvalue_scale_parameter = torch.Tensor([2.0])
shape_parameter = 2.0 * torch.ones(order)


"""
Set up the gammas
"""
initial_gamma_val = 6
gammas = initial_gamma_val * torch.ones(2 * order)
gammas[0] = 1

parameters = {
    "gammas": gammas,
    "noise_parameter": noise_parameter,
    "eigenvalue_smoothness_parameter": eigenvalue_smoothness_parameter,
    "eigenvalue_scale_parameter": eigenvalue_scale_parameter,
    "shape_parameter": shape_parameter,
}
gammas.requires_grad = False
noise_parameter.requires_grad = True
eigenvalue_smoothness_parameter.requires_grad = False
eigenvalue_scale_parameter.requires_grad = True
shape_parameter.requires_grad = True

basis = get_symmetric_orthonormal_basis(order, gammas)
optimiser = torch.optim.Adam(
    [value for value in parameters.values()], lr=0.001
)
likelihood = MercerLikelihood(
    order, optimiser, basis, input_sample, output_sample
)
likelihood.fit(parameters)

breakpoint()
favard_gp = build_favard_gp(parameters, order, input_sample, output_sample)
breakpoint()

fineness = 400
x = torch.linspace(-4, 4, fineness)
plt.scatter(input_sample, output_sample)
for i in range(10):
    sample_gp = favard_gp.gen_gp()
    plt.plot(x, sample_gp(x))
plt.show()

mean_gp = favard_gp.get_posterior_mean()

plt.plot(x, mean_gp(x), color="black")
plt.plot(x, test_function(x), color="red")
plt.show()
