import torch
import torch.distributions as D
from mercergp.MGP import MercerGP, MercerKernel, HilbertSpaceElement
from ortho.basis_functions import (
    Basis,
    smooth_exponential_basis_fasshauer,
    smooth_exponential_eigenvalues_fasshauer,
)
from ortho.orthopoly import OrthonormalPolynomial
from ortho.builders import get_orthonormal_basis
from mercergp.likelihood import FavardLikelihood

# from mercergp.builders import build_mercer_gp
from builders import build_favard_gp
import matplotlib.pyplot as plt


def target_function(x: torch.Tensor) -> torch.Tensor:
    return 6 * torch.sin(x)


# build the "ground truth"
gp_order = 6
sample_size = 200
fineness = 500
sample_shape = torch.Size([sample_size])
input_sample = D.Normal(0.0, 2.0).sample(sample_shape)
noise_sample = D.Normal(0.0, 0.5).sample(sample_shape)
output_sample = target_function(input_sample) + noise_sample

# true_coefficients = torch.Tensor([1, 2, 4, 7, 6, 2, 4, 1, 1, 1])

initial_gamma_val = 6
gammas = initial_gamma_val * torch.ones(2 * gp_order)
gammas[0] = 1

noise_parameter = torch.Tensor([0.5])
eigenvalue_smoothness_parameter = torch.Tensor([6.0])
eigenvalue_scale_parameter = torch.Tensor([2.0])
shape_parameter = 2.0 * torch.ones(gp_order)
noise_parameter.requires_grad = True
kernel_params = {
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

"""
Plotting parameters
"""
end_point = 6
x_axis = torch.linspace(-end_point, end_point, fineness)

"""
Build the basis and the Favard GP
"""
orthobasis = get_orthonormal_basis(gp_order, torch.zeros(2 * gp_order), gammas)
init_fgp = build_favard_gp(
    kernel_params, gp_order, input_sample, output_sample
)

"""
Plot some samples before training
"""
init_fgp.add_data(input_sample, output_sample)

sample_gp = init_fgp.gen_gp()
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
likelihood = FavardLikelihood(
    gp_order,
    optimiser,
    orthobasis,
    input_sample,
    output_sample,
)
likelihood.fit(kernel_params)


"""
Having trained the parameters, we can now build the Gaussian process.
"""
fgp = build_favard_gp(kernel_params, gp_order, input_sample, output_sample)
fgp.add_data(input_sample, output_sample)
func_count = 10
plt.scatter(
    input_sample.detach().numpy().flatten(),
    output_sample.detach().numpy().flatten(),
    marker="+",
)
for i in range(func_count):
    sample = fgp.gen_gp()
    plt.plot(x_axis, sample(x_axis))  # 1.3 seems to lead to a good fit??

mean = fgp.get_posterior_mean()
plt.plot(x_axis, mean(x_axis))
plt.show()
