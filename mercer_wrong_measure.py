import torch
import torch.distributions as D
from mercergp.MGP import MercerGP, MercerKernel, HilbertSpaceElement
from ortho.basis_functions import (
    Basis,
    smooth_exponential_basis_fasshauer,
    smooth_exponential_eigenvalues_fasshauer,
)
from mercergp.likelihood import MercerLikelihood
from mercergp.builders import build_mercer_gp
import matplotlib.pyplot as plt


def target_function(x: torch.Tensor) -> torch.Tensor:
    return 6 * torch.sin(x)


def get_input_sample(sample_size):
    """
    Returns a sample for the purpose of testing different input measures for
    the problem of the regression.
    """
    # sample = D.Normal(0.0, 1).sample([sample_size])
    std = torch.Tensor([1.0])
    mix = D.Categorical(torch.ones(2))
    comp = D.Normal(torch.Tensor([-6, 6]), torch.Tensor([std, std]))
    dist = torch.distributions.MixtureSameFamily(mix, comp)
    sample = dist.sample((sample_size,))
    # plt.hist(sample.numpy().flatten(), bins=20)
    # plt.show()
    return sample


def get_output_sample(input_sample, true_noise_parameter):
    sample_size = len(input_sample)
    output_sample = target_function(input_sample) + D.Normal(
        0.0, true_noise_parameter.squeeze()
    ).sample([sample_size])
    return output_sample


def get_samples(sample_size, true_noise_parameter):
    input_sample = get_input_sample(sample_size)
    return input_sample, get_output_sample(input_sample, true_noise_parameter)


# build the "ground truth"
order = 10
fineness = 500
end_point = 6

# true_coefficients = torch.Tensor([1, 2, 4, 7, 6, 2, 4, 1, 1, 1])
ard_parameter = torch.Tensor([[1.0]])
precision_parameter = torch.Tensor([1.0])
noise_parameter = torch.Tensor([1.0])
ard_parameter.requires_grad = True
precision_parameter.requires_grad = True
noise_parameter.requires_grad = False
kernel_params = {
    "ard_parameter": ard_parameter,
    "precision_parameter": precision_parameter,
    "noise_parameter": noise_parameter,
}

basis = Basis(smooth_exponential_basis_fasshauer, 1, order, kernel_params)
# target_function = HilbertSpaceElement(basis, true_coefficients)
x_axis = torch.linspace(-end_point, end_point, fineness)

# build input and output samples
gp_order = 8
sample_size = 200

sample_shape = torch.Size([sample_size])
# input_sample = D.Normal(0.0, 2.0).sample(sample_shape)
# noise_sample = D.Normal(0.0, 0.5).sample(sample_shape)
# output_sample = target_function(input_sample) + noise_sample
true_noise_parameter = torch.Tensor([0.5])
input_sample, output_sample = get_samples(sample_size, true_noise_parameter)

new_basis = Basis(
    smooth_exponential_basis_fasshauer, 1, gp_order, kernel_params
)
eigenvalues = smooth_exponential_eigenvalues_fasshauer(gp_order, kernel_params)
kernel = MercerKernel(gp_order, new_basis, eigenvalues, kernel_params)
init_mgp = MercerGP(new_basis, gp_order, 1, kernel)
init_mgp.add_data(input_sample, output_sample)

sample_gp = init_mgp.gen_gp()
plt.plot(x_axis.detach().numpy(), sample_gp(x_axis).detach().numpy())
plt.scatter(
    input_sample.detach().numpy().flatten(),
    output_sample.detach().numpy().flatten(),
    marker="+",
)
plt.show()

"""
Build the optimiser and likelihood, and train the parameters
"""
optimiser = torch.optim.Adam(
    [param for param in kernel_params.values()], lr=0.01
)
likelihood = MercerLikelihood(
    order,
    optimiser,
    new_basis,
    input_sample,
    output_sample,
    eigenvalue_generator=lambda params: smooth_exponential_eigenvalues_fasshauer(
        gp_order, params
    ),
)
likelihood.fit(kernel_params)


"""
Having trained the parameters, we can now build the Gaussian process.
"""
fresh_kernel_params = {
    "ard_parameter": ard_parameter.detach(),
    "precision_parameter": precision_parameter.detach(),
    "noise_parameter": noise_parameter.detach(),
}

cleanbasis = Basis(
    smooth_exponential_basis_fasshauer, 1, gp_order, fresh_kernel_params
)
mgp = build_mercer_gp(
    cleanbasis,
    ard_parameter.detach(),
    precision_parameter.detach(),
    noise_parameter.detach(),
    gp_order,
    1,
)
mgp.add_data(input_sample.detach(), output_sample.detach())
func_count = 10
plt.scatter(
    input_sample.detach().numpy().flatten(),
    output_sample.detach().numpy().flatten(),
    marker="+",
)
for i in range(func_count):
    sample = mgp.gen_gp()
    plt.plot(
        x_axis, sample(x_axis).detach()
    )  # 1.3 seems to lead to a good fit??

mean = mgp.get_posterior_mean()
plt.plot(x_axis, mean(x_axis).detach())
plt.show()
