import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import tikzplotlib

"""
This script conducts the analysis for experiment 12, which generates the 
 predictive density for the true GP, Mercer GP and Favard
GP. Then, it calculates the kl-divergence:
    - from Mercer GP predictive density to true GP predictive density
    - from Favard GP predictive density to true GP predictive density

These are stored in a .pt tensor file, with a tensor of shape
[experiment_count, mixture_count, 2].

experiment_count is 1000; mixture_count is 5.

mixture_count represents the travel from a Normal distribution to an 
increasingly "mixed" input distribution. One would expect that the 
KL divergence across mixture_count would increase, but _less_ for the Favard.
"""

normal_input_kls = torch.load(
    "./experiment_12_data/kl_divergences_normal_test_inputs.pt"
)
mixture_input_kls = torch.load(
    "./experiment_12_data/kl_divergences_mixture_test_inputs.pt"
)

# print(kls.shape)
print(normal_input_kls)
print(mixture_input_kls)
# results = torch.mean(kls, dim=0)
# print(results)
# breakpoint()
first_plot = False
second_plot = False

experiment_count = normal_input_kls.shape[0]
if first_plot:
    for i in range(experiment_count):
        graph_0 = normal_input_kls[i, :, 0]
        graph_1 = normal_input_kls[i, :, 1]
        # breakpoint()
        loggraph_0 = torch.mean(torch.log(graph_0))
        loggraph_1 = torch.mean(torch.log(graph_1))
        plt.plot(loggraph_0 - loggraph_1)
        # plt.plot(graph_1 - graph_0)
        # plt.plot(graph_1)
    plt.show()

if second_plot:
    for i in range(experiment_count):
        graph_0 = mixture_input_kls[i, :, 0]
        graph_1 = mixture_input_kls[i, :, 1]
        plt.plot(torch.log(graph_1) - torch.log(graph_0))
        # plt.plot(graph_1)
    plt.show()

log_data = False
end_order = 5

# now show the graphs with the mean
if log_data:
    normal_meangraph_0 = torch.mean(
        torch.log(normal_input_kls[:, :end_order, 0]), dim=0
    )
    normal_meangraph_1 = torch.mean(
        torch.log(normal_input_kls[:, :end_order, 1]), dim=0
    )
    normal_lqgraph_0 = torch.quantile(
        torch.log(normal_input_kls[:, :end_order, 0]), 0.25, dim=0
    )
    normal_hqgraph_0 = torch.quantile(
        torch.log(normal_input_kls[:, :end_order, 0]), 0.75, dim=0
    )
    normal_lqgraph_1 = torch.quantile(
        torch.log(normal_input_kls[:, :end_order, 1]), 0.25, dim=0
    )
    normal_hqgraph_1 = torch.quantile(
        torch.log(normal_input_kls[:, :end_order, 1]), 0.75, dim=0
    )
    mixture_meangraph_0 = torch.mean(
        torch.log(mixture_input_kls[:, :end_order, 0]), dim=0
    )
    mixture_meangraph_1 = torch.mean(
        torch.log(mixture_input_kls[:, :end_order, 1]), dim=0
    )
    mixture_lqgraph_0 = torch.quantile(
        torch.log(mixture_input_kls[:, :end_order, 0]), 0.25, dim=0
    )
    mixture_hqgraph_0 = torch.quantile(
        torch.log(mixture_input_kls[:, :end_order, 0]), 0.75, dim=0
    )
    mixture_lqgraph_1 = torch.quantile(
        torch.log(mixture_input_kls[:, :end_order, 1]), 0.25, dim=0
    )
    mixture_hqgraph_1 = torch.quantile(
        torch.log(mixture_input_kls[:, :end_order, 1]), 0.75, dim=0
    )
else:
    normal_meangraph_0 = torch.mean(normal_input_kls[:, :end_order, 0], dim=0)
    normal_meangraph_1 = torch.mean(normal_input_kls[:, :end_order, 1], dim=0)
    normal_lqgraph_0 = torch.quantile(normal_input_kls[:, :end_order, 0], 0.25, dim=0)
    normal_hqgraph_0 = torch.quantile(normal_input_kls[:, :end_order, 0], 0.75, dim=0)
    normal_lqgraph_1 = torch.quantile(normal_input_kls[:, :end_order, 1], 0.25, dim=0)
    normal_hqgraph_1 = torch.quantile(normal_input_kls[:, :end_order, 1], 0.75, dim=0)
    mixture_meangraph_0 = torch.mean(mixture_input_kls[:, :end_order, 0], dim=0)
    mixture_meangraph_1 = torch.mean(mixture_input_kls[:, :end_order, 1], dim=0)
    mixture_lqgraph_0 = torch.quantile(mixture_input_kls[:, :end_order, 0], 0.25, dim=0)
    mixture_hqgraph_0 = torch.quantile(mixture_input_kls[:, :end_order, 0], 0.75, dim=0)
    mixture_lqgraph_1 = torch.quantile(mixture_input_kls[:, :end_order, 1], 0.25, dim=0)
    mixture_hqgraph_1 = torch.quantile(mixture_input_kls[:, :end_order, 1], 0.75, dim=0)
# normal_meangraph_0 = torch.mean((normal_input_kls[:, :, 0]), dim=0)
# normal_meangraph_1 = torch.mean((normal_input_kls[:, :, 1]), dim=0)

# normal distributed input test

plt.rcParams["text.usetex"] = True
# plt.rcParams["figure.figsize"] = (2, 1)
fig, ax = plt.subplots()
x = torch.linspace(0, len(normal_meangraph_0), len(normal_meangraph_0))


# ax.legend(
# (
# "Mean KL divergence with Mercer representation",
# "Mean KL divergence with Favard representation",
# )
# )
ax.set_xlabel(r"$i$")
ax.set_ylabel(r"$\mathcal{KL}[p_i||q]$")
ax.plot(
    x,
    normal_meangraph_0,
    x,
    normal_meangraph_1,
)
ax.plot(
    x,
    normal_lqgraph_0,
    x,
    normal_hqgraph_0,
    x,
    normal_lqgraph_1,
    x,
    normal_hqgraph_1,
    linestyle="--",
)
tikzplotlib.save(
    "/home/william/phd/tex_projects/favard_kernels_icml/diagrams/predictivedensitydiagram1.tex",
    axis_height="\\predictivedensitydiagramheight",
    axis_width="\\predictivedensitydiagramwidth",
)
# plt.savefig(
# "/home/william/phd/tex_projects/favard_kernels_icml/diagrams/predictivedensitydiagram.eps",
# format="eps",
# dpi=1200,
# )
fig2, ax2 = plt.subplots()
ax2.set_xlabel(r"$i$")
ax2.set_ylabel(r"$\mathcal{KL}[p_i||q]$")
ax2.plot(
    x,
    mixture_meangraph_0,
    x,
    mixture_meangraph_1,
)
ax2.plot(
    x,
    mixture_lqgraph_0,
    x,
    mixture_hqgraph_0,
    x,
    mixture_lqgraph_1,
    x,
    mixture_hqgraph_1,
    linestyle="--",
)
tikzplotlib.save(
    "/home/william/phd/tex_projects/favard_kernels_icml/diagrams/predictivedensitydiagram2.tex",
    axis_height="\\predictivedensitydiagramheight",
    axis_width="\\predictivedensitydiagramwidth",
)
# plt.savefig(
# "/home/william/phd/tex_projects/favard_kernels_icml/diagrams/predictivedensitydiagram2.eps",
# format="eps",
# dpi=1200,
# )
# plt.savefig(
# "/home/william/phd/tex_projects/favard_kernels_icml/diagrams/predictivedensitydiagram.svg",
# format="svg",
# dpi=1200,
# )
# plt.show()
# ax.set_title()

# plt.plot(normal_meangraph_0.numpy())
# plt.plot(normal_lqgraph_0.numpy(), linestyle="--")
# plt.plot(normal_hqgraph_0.numpy(), linestyle="--")
# plt.plot(normal_meangraph_1.numpy())
# plt.plot(normal_lqgraph_1.numpy(), linestyle="--")
# plt.plot(normal_hqgraph_1.numpy(), linestyle="--")
# # plt.ylabel(r"KL[p_i || q]")
# # plt.xlabel(r"i")
# plt.show()

# mixture input test
# plt.plot(mixture_meangraph_0.numpy())
# plt.plot(mixture_lqgraph_0.numpy(), linestyle="--")
# plt.plot(mixture_hqgraph_0.numpy(), linestyle="--")
# plt.plot(mixture_meangraph_1.numpy())
# plt.plot(mixture_lqgraph_1.numpy(), linestyle="--")
# plt.plot(mixture_hqgraph_1.numpy(), linestyle="--")
# plt.ylabel(r"\mathcal{KL}\left[g_i || q\right]")
# plt.xlabel(r"i")
# plt.show()
