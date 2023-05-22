import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import ortho.basis_functions as bf

from mercergp.MGP import MercerGP, MercerGPFourierPosterior
from mercergp.posterior_sampling import histogram_spectral_distribution
from mercergp.kernels import MercerKernel
from mercergp.eigenvalue_gen import SmoothExponentialFasshauer
from mercergp.posterior_sampling import histogram_spectral_distribution
from mercergp.builders import (
    build_mercer_gp_fourier_posterior,
    build_mercer_gp,
)
from favard_kernels.builders import build_favard_gp
import tikzplotlib


def test_function(x: torch.Tensor) -> torch.Tensor:
    """
    Test function used in an iteration of Daskalakis, Dellaportas and Panos.
    """
    return (1.5 * torch.sin(x) + 0.5 * torch.cos(4 * x) + x / 8).squeeze()


# plotting
save_tikz = False
axis_width = 10
x_axis = torch.linspace(-axis_width, axis_width, 1000)  # .unsqueeze(1)
test_sample_size = 2000
test_sample_shape = torch.Size([test_sample_size])
marker_value = "."
marker_size = 0.5

# hyperparameters
order = 8
dimension = 1
l_se = torch.Tensor([[2.2]])
sigma_se = torch.Tensor([2.0])
prec = torch.Tensor([1.0])
sigma_e = torch.Tensor([0.3])
kernel_args = {
    "ard_parameter": l_se,
    "variance_parameter": sigma_se,
    "noise_parameter": sigma_e,
    "precision_parameter": prec,
}

"""
Part 1: Some random samples that exhibit the variance starvation
"""
# inputs and outputs
inputs = D.Normal(0.0, 1.0).sample(test_sample_shape)
outputs = test_function(inputs) + sigma_e * D.Normal(0.0, 1.0).sample(
    inputs.shape
)
eigenvalue_generator = SmoothExponentialFasshauer(order)
basis = bf.Basis(
    bf.smooth_exponential_basis_fasshauer,
    dimension,
    order,
    kernel_args,
)
print("about to build the mercer gp")
mercer_gp = build_favard_gp(
    kernel_args,
    order,
    inputs,
    lambda x: torch.exp(-(x**2)),
    # lambda x: torch.ones(x.shape),
    eigenvalue_generator,
)
generated_basis = mercer_gp.basis
mercer_gp.add_data(inputs, outputs)
# plt.plot(x_axis, generated_basis(x_axis))
# plt.plot(x_axis, basis(x_axis), color="red")
# plt.show()
# breakpoint()
plt.rcParams["text.usetex"] = True
fig, ax2 = plt.subplots()
ax2.set_xlabel(r"$\mathcal{X}$")
ax2.set_ylabel(r"Function Values")
ax2.scatter(inputs, outputs, marker=marker_value, s=marker_size)
for i in range(5):
    sample = mercer_gp.gen_gp()
    ax2.plot(
        x_axis,
        sample(x_axis),
    )
posterior_mean = mercer_gp.get_posterior_mean()
ax2.plot(x_axis, posterior_mean(x_axis), color="black")
# plt.plot(x_axis, sample(x_axis))
# plt.scatter(inputs, outputs, marker="+")
# plt.show()

# ax2.legend(
# (
# "Ground Truth",
# "Re(Posterior sample)",
# # "Im(Posterior sample)",
# "Posterior mean",
# )
# )
if save_tikz:
    tikzplotlib.save(
        "/home/william/phd/tex_projects/favard_kernels_icml/diagrams/posteriorsample1.tex",
        axis_height="\\posteriorsamplediagramheight",
        axis_width="\\posteriorsamplediagramwidth",
    )
else:
    plt.show()
# plt.savefig(
# "/home/william/phd/tex_projects/favard_kernels_icml/diagrams/posteriorsamples1.eps",
# format="eps",
# dpi=1200,
# )
# plt.show()
"""
Part 2: Posterior mean and a posterior sample
"""
order = 8
rff_order = 4700
rff_frequency = 1000
dimension = 1
l_se = torch.Tensor([[1.70]])
sigma_se = torch.Tensor([2.0])
prec = torch.Tensor([1.0])
sigma_e = torch.Tensor([0.3])
kernel_args = {
    "ard_parameter": l_se,
    "variance_parameter": sigma_se,
    "noise_parameter": sigma_e,
    "precision_parameter": prec,
}
# basis = Basis()

basis = bf.Basis(
    bf.smooth_exponential_basis_fasshauer,
    dimension,
    order,
    kernel_args,
)
eigenvalue_generator = SmoothExponentialFasshauer(order)
# mercer
# mercer with fourter posterior
print("about to build the fourier posterior gp")
mercer_gp_fourier_posterior = build_mercer_gp_fourier_posterior(
    kernel_args,
    order,
    basis,
    eigenvalue_generator,
    frequency=rff_frequency,
    spectral_distribution_type="gaussian",
    rff_order=rff_order,
)

# mercer_gp.add_data(inputs, outputs)

mercer_gp_fourier_posterior.add_data(inputs, outputs)
posterior_mean = mercer_gp_fourier_posterior.get_posterior_mean()
posterior_sample_data = posterior_mean(x_axis)
fourier_posterior_sample = mercer_gp_fourier_posterior.gen_gp()
fourier_sample_data = fourier_posterior_sample(x_axis)
true_function = test_function(x_axis)

plt.rcParams["text.usetex"] = True
# plt.rcParams["figure.figsize"] = (6, 4)
fig, ax = plt.subplots()

ax.set_xlabel(r"$\mathcal{X}$")
ax.set_ylabel(r"Function Values")
ax.scatter(inputs, outputs, marker=marker_value, s=marker_size)
ax.plot(
    x_axis,
    true_function,
    x_axis,
    fourier_sample_data.real,
    # x_axis,
    # fourier_sample_data.imag,
    x_axis,
    posterior_sample_data,
)
ax.legend(
    (
        "Ground Truth",
        "Re(Posterior sample)",
        # "Im(Posterior sample)",
        "Posterior mean",
    )
    # fontsize="x-small",
)

# plt.axvline(x=true_order, color="r", linestyle="--")
if save_tikz:
    tikzplotlib.save(
        "/home/william/phd/tex_projects/favard_kernels_icml/diagrams/posteriorsample2.tex",
        axis_height="\\posteriorsamplediagramheight",
        axis_width="\\posteriorsamplediagramwidth",
    )
else:
    plt.show()
# plt.savefig(
# "/home/william/phd/tex_projects/favard_kernels_icml/diagrams/posteriorsample.eps",
# format="eps",
# dpi=1200,
# )
# plt.plot(x_axis, fourier_sample_data.real)
# # if fourier_sample_data.is_complex():
# # plt.plot(x_axis, fourier_sample_data.imag)
# # plt.plot(x_axis, fourier_posterior_sample.posterior_component(x_axis), color="red")
# plt.scatter(inputs, outputs)
# plt.plot(x_axis, posterior_sample_data, color="green")
# plt.plot(x_axis, test_function(x_axis), color="magenta")

plt.show()
