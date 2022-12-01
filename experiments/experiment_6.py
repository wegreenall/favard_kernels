import torch
import torch.distributions as D

from mercergp.builders import (
    train_smooth_exponential_mercer_params,
    build_mercer_gp,
)
from mercergp.eigenvalue_gen import SmoothExponentialFasshauer
import matplotlib.pyplot as plt

"""
This experiment shows a synthetic example of the Mercer GP construction, with
incorrect input measure.

This should yield a reasonable orthgonal basis given the input measure,
and it likely will not be far from the Hermite functions.
"""


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
true_noise_parameter = torch.Tensor([0.1])
noise_parameter = torch.Tensor([2.0])
ard_parameter = torch.Tensor([1.0])
precision_parameter = torch.Tensor([1.0])
noise_parameter.requires_grad = True
precision_parameter.requires_grad = False
ard_parameter.requires_grad = True
parameters = {
    "noise_parameter": noise_parameter,
    "ard_parameter": ard_parameter,
    "precision_parameter": precision_parameter,
}
order = 10
# eigenvalue_generator = SmoothExponentialFasshauer(order)

# input_sample
sample_size = 400
sample_shape = torch.Size([sample_size])
noise_sample = (
    D.Normal(0.0, true_noise_parameter).sample(sample_shape).squeeze()
)
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
    [param for param in parameters.values()], lr=0.001
)

fitted_parameters = train_smooth_exponential_mercer_params(
    parameters,
    order,
    input_sample,
    output_sample,
    optimiser,
)

mercer_gp = build_mercer_gp(
    fitted_parameters,
    order,
    input_sample,
    output_sample,
)
mercer_gp.add_data(input_sample, output_sample)

x_axis = torch.linspace(-6, 6, 1000)
for i in range(5):
    gp_sample = mercer_gp.gen_gp()
    plt.plot(x_axis, gp_sample(x_axis))

plt.plot(x_axis, test_function(x_axis), ls="dashed")
plt.scatter(input_sample, output_sample)
plt.show()
