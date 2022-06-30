import torch
import torch.distributions as D
from favard_kernels.builders import build_favard_gp

from mercergp.builders import train_mercer_params, build_mercer_gp
from mercergp.MGP import SmoothExponentialRFFGP
from mercergp.eigenvalue_gen import SmoothExponentialFasshauer
import matplotlib.pyplot as plt

"""
This script will attempt to build an example of the machinery for constructing 
the posterior Gaussian process as a decoupled basis. Following "Wilson et al 2020",
which itself appears to follow "Hensman et al 2017" relatively closely, the aim 
is to evaluate the process of essentially modelling the posterior 'component' 
using a sparse GP, whilst keeping the prior "component" in the form of the
RFF setup.
"""

# first, generate the data.


def weight_function(x: torch.Tensor) -> torch.Tensor:
    """
    Evaluates the weight function. Usually, e^{-lx^2}
    """
    length = torch.tensor(1.5)
    return torch.exp(-length * x ** 2)


def test_function(x: torch.Tensor) -> torch.Tensor:
    """
    The test function used in an iteration of Daskalakis, Dellaportas and Panos.
    """
    return (1.5 * torch.sin(x) + 0.5 * torch.cos(4 * x) + x / 8).squeeze()


"""
 required inputs for "train_favard_params":
  - parameters: dict,
  - eigenvalue_generator: EigenvalueGenerator,
  - order: int,
  - input_sample: torch.Tensor,
  - output_sample: torch.Tensor,
  - weight_function: Callable,
  - optimiser: torch.optim.Optimizer,
 The parameters dictionary should include the appropriate parameters as noted
 in the corresponding eigenvalue generator class.
"""
true_noise_parameter = torch.Tensor([0.2])
noise_parameter = torch.Tensor([0.2])
ard_parameter = torch.Tensor([1.0])
precision_parameter = torch.Tensor([2.00])
noise_parameter.requires_grad = False
precision_parameter.requires_grad = False
ard_parameter.requires_grad = False
parameters = {
    "noise_parameter": noise_parameter,
    "ard_parameter": ard_parameter,
    "precision_parameter": precision_parameter,
}
order = 10
eigenvalue_generator = SmoothExponentialFasshauer(order)


"""
Construct the input/output samples
"""
sample_size = 400
sample_shape = torch.Size([sample_size])
noise_sample = D.Normal(0.0, true_noise_parameter).sample(sample_shape).squeeze()
mean_vector = torch.Tensor([-0.0, 3.0])
variance_vector = torch.Tensor([1.0, 1.0])
core_dist = D.Normal(mean_vector, variance_vector)
mixture_dist = D.Categorical(torch.Tensor([0.2, 0.8]))
# mixture_dist = D.Categorical(torch.Tensor([1.0, 0.0]))
input_dist = D.MixtureSameFamily(mixture_dist, core_dist)
input_sample = input_dist.sample(sample_shape)
# input_sample = D.Normal(0.0, 1.0).sample(sample_shape)

rff_gp = SmoothExponentialRFFGP(500 * order, 1)
# to build this, we just take the output sample to be the residuals from the
# prior function.
output_sample = test_function(input_sample) + noise_sample

optimiser = torch.optim.Adam(
    [param for param in parameters.values() if isinstance(param, torch.Tensor)],
    lr=0.001,
)
favard_gp = build_favard_gp(
    parameters,
    eigenvalue_generator,
    order,
    input_sample,
    output_sample,
    weight_function,
    dim,
)
# mercer_gp.add_data(input_sample, residual_sample)

x_axis = torch.linspace(-9, 9, 1000)
for i in range(20):
    rff_sample = rff_gp.gen_gp()
    residual_sample = output_sample - rff_sample(input_sample)
    favard_gp.set_data(input_sample, residual_sample)
    posterior_sample = favard_gp.gen_gp()
    plt.plot(x_axis, posterior_sample(x_axis) + rff_sample(x_axis))
# plt.show()

# for i in range(5):
# gp_sample = mercer_gp.gen_gp()
# plt.plot(x_axis, gp_sample(x_axis) + genned_prior(x_axis))

plt.plot(x_axis, test_function(x_axis), ls="dashed")
plt.scatter(input_sample, output_sample)
plt.show()
