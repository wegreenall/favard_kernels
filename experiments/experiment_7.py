import torch
import torch.distributions as D
import matplotlib.pyplot as plt

from mercergp.builders import train_mercer_params, build_mercer_gp
from mercergp.eigenvalue_gen import SmoothExponentialFasshauer

"""
This experiment will compare the predictive density found under the Hermite 
function Mercer expansion with a non-Gaussian input distribution, vs a 
Gaussian input distribution.
"""


def test_function(x: torch.Tensor) -> torch.Tensor:
    """
    Test function used in an iteration of Daskalakis, Dellaportas and Panos.
    """
    return (1.5 * torch.sin(x) + 0.5 * torch.cos(4 * x) + x / 8).squeeze()


# set up the parameters
true_noise_parameter = torch.Tensor([0.1])
gaussian_noise_parameter = torch.Tensor([2.0])
gaussian_ard_parameter = torch.Tensor([1.0])
gaussian_precision_parameter = torch.Tensor([1.0])
gaussian_noise_parameter.requires_grad = True
gaussian_precision_parameter.requires_grad = False
gaussian_ard_parameter.requires_grad = True
gaussian_parameters = {
    "noise_parameter": gaussian_noise_parameter,
    "ard_parameter": gaussian_ard_parameter,
    "precision_parameter": gaussian_precision_parameter,
}
order = 15
eigenvalue_generator = SmoothExponentialFasshauer(order)

# sample data construction
sample_size = 400
sample_shape = torch.Size([sample_size])
noise_dist = D.Normal(0.0, true_noise_parameter)
noise_sample = noise_dist.sample(sample_shape).squeeze()

# Gaussian input sample
gaussian_input_sample = D.Normal(0.0, 1.0).sample(sample_shape)
gaussian_output_sample = test_function(gaussian_input_sample) + noise_sample

optimiser = torch.optim.Adam(
    [param for param in gaussian_parameters.values()], lr=0.001
)

# train the model
fitted_parameters = train_mercer_params(
    gaussian_parameters,
    order,
    gaussian_input_sample,
    gaussian_output_sample,
    optimiser,
)

gaussian_mercer_gp = build_mercer_gp(
    fitted_parameters,
    order,
    gaussian_input_sample,
    gaussian_output_sample,
)
gaussian_mercer_gp.add_data(gaussian_input_sample, gaussian_output_sample)

x_axis = torch.linspace(-3, 3, 1000)
for i in range(5):
    gp_sample = gaussian_mercer_gp.gen_gp()
    plt.plot(x_axis, gp_sample(x_axis))

plt.plot(x_axis, test_function(x_axis), ls="dashed")
plt.scatter(gaussian_input_sample, gaussian_output_sample)
plt.show()

test_sample_size = 50
test_sample_shape = torch.Size([test_sample_size])
test_points = D.Normal(0.0, 1.0).sample(test_sample_shape)
gaussian_predictive_density = gaussian_mercer_gp.get_predictive_density(
    test_points
)

"""
Build the non-Gaussian input distribution Mercer GP
"""

# now build the same thing for non-Gaussian inputs
# set up the parameters
non_gaussian_noise_parameter = torch.Tensor([2.0])
non_gaussian_ard_parameter = torch.Tensor([1.0])
non_gaussian_precision_parameter = torch.Tensor([1.0])
non_gaussian_noise_parameter.requires_grad = True
non_gaussian_precision_parameter.requires_grad = False
non_gaussian_ard_parameter.requires_grad = True
non_gaussian_parameters = {
    "noise_parameter": non_gaussian_noise_parameter,
    "ard_parameter": non_gaussian_ard_parameter,
    "precision_parameter": non_gaussian_precision_parameter,
}
order = 15
eigenvalue_generator = SmoothExponentialFasshauer(order)

# sample data construction
sample_size = 400
sample_shape = torch.Size([sample_size])
noise_sample = (
    D.Normal(0.0, true_noise_parameter).sample(sample_shape).squeeze()
)

# non-Gaussian input sample
mixing_distribution = D.Categorical(torch.Tensor([0.2, 0.8]))
component_distribution = D.Normal(torch.Tensor([-3, 3]), torch.Tensor([1, 1]))
non_gaussian_input_distribution = D.MixtureSameFamily(
    mixing_distribution, component_distribution
)
non_gaussian_input_sample = non_gaussian_input_distribution.sample(
    sample_shape
)
non_gaussian_output_sample = (
    test_function(non_gaussian_input_sample) + noise_sample
)

optimiser = torch.optim.Adam(
    [param for param in non_gaussian_parameters.values()], lr=0.001
)

# train the model
non_gaussian_fitted_parameters = train_mercer_params(
    non_gaussian_parameters,
    order,
    non_gaussian_input_sample,
    non_gaussian_output_sample,
    optimiser,
)

non_gaussian_mercer_gp = build_mercer_gp(
    non_gaussian_fitted_parameters,
    order,
    non_gaussian_input_sample,
    non_gaussian_output_sample,
)
non_gaussian_mercer_gp.add_data(
    non_gaussian_input_sample, non_gaussian_output_sample
)

x_axis = torch.linspace(-3, 3, 1000)
for i in range(5):
    gp_sample = non_gaussian_mercer_gp.gen_gp()
    plt.plot(x_axis, gp_sample(x_axis))

plt.plot(x_axis, test_function(x_axis), ls="dashed")
plt.scatter(non_gaussian_input_sample, non_gaussian_output_sample)
plt.show()

test_sample_size = 50
test_sample_shape = torch.Size([test_sample_size])
# test_points = D.Normal(0.0, 1.0).sample(test_sample_shape)
test_points = torch.Tensor([0.0])
gaussian_predictive_density = gaussian_mercer_gp.get_predictive_density(
    test_points
)
non_gaussian_predictive_density = (
    non_gaussian_mercer_gp.get_predictive_density(test_points)
)

print("gaussian predictive density:", gaussian_predictive_density)
print("non gaussian predictive density:", non_gaussian_predictive_density)

# get some samples from the function at the "truth".
mean = test_function(torch.Tensor([0.0]))
true_sample = mean + noise_dist.sample(test_sample_shape)
true_test_dist = D.Normal(mean, true_noise_parameter)

# breakpoint()
non_gaussian_kl = torch.distributions.kl_divergence(
    non_gaussian_predictive_density, true_test_dist
)
gaussian_kl = torch.distributions.kl_divergence(
    gaussian_predictive_density, true_test_dist
)

print("gaussian kl", gaussian_kl)
print("non gaussian kl", non_gaussian_kl)
