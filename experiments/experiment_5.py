import torch
import torch.distributions as D

from favard_kernels.builders import train_favard_params, build_favard_gp
from mercergp.eigenvalue_gen import (
    SmoothExponentialFasshauer,
    PolynomialEigenvalues,
)
import matplotlib.pyplot as plt

"""
This experiment shows a synthetic example of the Favard GP construction.

In this case, the input distribution is a Gaussian mixture (see lines
70 - 78). This is the Favard version, so it should work "better"
than the standard Mercer version.
"""

print(
    "This experiment shows a synthetic example of the Favard GP construction.\
This should yield a reasonable orthgonal basis given the input measure,\
and it likely will not be far from the Hermite functions."
)


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
order = 10
true_noise_parameter = torch.Tensor([0.1])
noise_parameter = torch.Tensor([2.0])
ard_parameter = torch.Tensor([1.0])
precision_parameter = torch.ones(order)
noise_parameter.requires_grad = True
precision_parameter.requires_grad = True
ard_parameter.requires_grad = True
parameters = {
    "noise_parameter": noise_parameter,
    "scale": ard_parameter,
    "shape": precision_parameter,
    "degree": order,
}
eigenvalue_generator = PolynomialEigenvalues(order)

# input_sample
sample_size = 400
sample_shape = torch.Size([sample_size])
noise_sample = (
    D.Normal(0.0, true_noise_parameter).sample(sample_shape).squeeze()
)
# input_sample = D.Normal(0.0, 1.0).sample(sample_shape)
mixture_dist = D.Categorical(torch.Tensor([0.2, 0.8]))
means = 1.4 * torch.Tensor([-1.0, 3.0])
variances = torch.Tensor([1.0, 1.0])
component_dist_1 = D.Normal(means, variances)


favard_input_measure = D.MixtureSameFamily(mixture_dist, component_dist_1)
input_sample = favard_input_measure.sample(sample_shape)
output_sample = test_function(input_sample) + noise_sample

# optimiser = torch.optim.SGD(
# [param for param in parameters.values()], lr=0.00001
# )
optimiser = torch.optim.Adam(
    [
        param
        for param in parameters.values()
        if isinstance(param, torch.Tensor)
    ],
    lr=0.001,
)

fitted_parameters = train_favard_params(
    parameters,
    eigenvalue_generator,
    order,
    input_sample,
    output_sample,
    weight_function,
    optimiser,
)

favard_gp = build_favard_gp(
    fitted_parameters,
    eigenvalue_generator,
    order,
    input_sample,
    output_sample,
    weight_function,
)
favard_gp.add_data(input_sample, output_sample)

x_axis = torch.linspace(-6, 6, 1000)
for i in range(20):
    gp_sample = favard_gp.gen_gp()
    plt.plot(x_axis, gp_sample(x_axis))

plt.plot(x_axis, test_function(x_axis), ls="dashed")
plt.scatter(input_sample, output_sample)
plt.show()
