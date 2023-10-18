import torch
from favard_kernels.datasets.experiment_utils import (
    DataGenerator,
    Experiment,
    KernelType,
    GPType,
)
import pickle
import matplotlib.pyplot as plt
from scipy.stats import kstest
import latextable
import texttable as tt
import tikzplotlib


filename = "./data/pit_stops.csv"

# get the data and clean it on "milliseconds
data_generator = DataGenerator(filename, "milliseconds")

"""
1) get the data
2) get the parameters trained or loaded
3) get GPs
4) get the predictive densities at subsamples
"""
input_sample, output_sample = data_generator.get_data(
    "milliseconds", "stop", standardise=True
)

# get the parameters
experiment = Experiment()


# hyperparameters
pretrained = True
precompared = True
save_tikz = True
order = 10
empirical_experiment_count = 1000
test_input_count = 100
input_count = 800

# (
# mercer_trained_parameters,
# mercer_trained_noise,
# ) = experiment.get_parameters(
# order, input_sample, output_sample, KernelType.MERCER
# )

# (
# favard_trained_parameters,
# favard_trained_noise,
# ) = experiment.get_parameters(
# order, input_sample, output_sample, KernelType.FAVARD
# )
if not pretrained:
    # calculate the trained parameters
    (
        mercer_trained_parameters,
        mercer_trained_noise,
    ) = experiment.get_parameters(
        order, input_sample, output_sample, KernelType.MERCER
    )
    (
        favard_trained_parameters,
        favard_trained_noise,
    ) = experiment.get_parameters(
        order, input_sample, output_sample, KernelType.FAVARD
    )

    # now save the parameters
    torch.save(
        mercer_trained_noise,
        "mercer_trained_noise.pt",
    )
    torch.save(
        favard_trained_noise,
        "favard_trained_noise.pt",
    )
    with open(
        "favard_trained_parameters.pkl",
        "wb",
    ) as f:
        pickle.dump(favard_trained_parameters, f)
    with open(
        "mercer_trained_parameters.pkl",
        "wb",
    ) as f:
        pickle.dump(mercer_trained_parameters, f)
else:
    # get the saved parameters
    with open(
        "favard_trained_parameters.pkl",
        "rb",
    ) as f:
        favard_trained_parameters = pickle.load(f)
    with open(
        "mercer_trained_parameters.pkl",
        "rb",
    ) as f:
        mercer_trained_parameters = pickle.load(f)

    # load the parameters
    mercer_trained_noise = torch.load("mercer_trained_noise.pt")
    favard_trained_noise = torch.load("favard_trained_noise.pt")

# get the GPs
mercer_gp = experiment.get_GP(
    order,
    mercer_trained_parameters,
    mercer_trained_noise,
    input_sample,
    output_sample,
    GPType.STANDARD,
    KernelType.MERCER,
)
favard_gp = experiment.get_GP(
    order,
    favard_trained_parameters,
    favard_trained_noise,
    input_sample,
    output_sample,
    GPType.STANDARD,
    KernelType.FAVARD,
)

input_mean = data_generator.input_mean
input_std = data_generator.input_std
# experiment.present_gp(mercer_gp, input_mean, input_std)
# experiment.present_gp(favard_gp, input_mean, input_std)

# get the predictive densities
if not precompared:
    (
        favard_predictive_densities,
        mercer_predictive_densities,
    ) = experiment.compare_gps(
        favard_gp,
        mercer_gp,
        input_sample,
        output_sample,
        empirical_experiment_count,
        test_input_count,
        input_count,
    )
    torch.save(
        torch.Tensor([favard_predictive_densities])
        - torch.Tensor([mercer_predictive_densities]),
        "density_diffs.pt",
    )
density_diffs = torch.load("density_diffs.pt")
stats = []
p_values = []

ks_test = kstest(input_sample, "norm")
stats.append(ks_test[0])
p_values.append(ks_test[1])
print(stats)
print(p_values)
table = tt.Texttable()
table.set_precision(10)
table.set_cols_align(["c", "c"])
table.set_cols_valign(["m", "m"])
table.add_rows(
    [
        ["KS Statistic", "p-value"],
        [stats[0], p_values[0]],
    ]
)
table_tex = latextable.draw_latex(
    table, caption="KS Test Results", label="tab:formula_1_ks_test"
)
# write table tex to file
with open("formula_1_ks.tex", "w") as f:
    f.write(table_tex)

# histogram plotting
fig, ax = plt.subplots()
plt.rcParams["text.usetex"] = True
# plt.rcParams["figure.figsize"] = (6, 4)
ax.set_xlabel(r"$\densityfavard - \densitymercer$")
ax.set_ylabel(r"Counts ")
plt.hist(
    density_diffs,
    bins=150,
)
# ax.scatter(inputs, outputs, marker=marker_value, s=marker_size)
# plt.axvline(x=true_order, color="r", linestyle="--", linewidth=0.5)
# ax.legend(
# (
# "Ground Truth",
# "Re(Posterior sample)",
# # "Im(Posterior sample)",
# "Posterior mean",
# )
# # fontsize="x-small",
# )
if save_tikz:
    tikzplotlib.save(
        "/home/william/phd/tex_projects/favard_kernels_neurips/diagrams/formula_1_dataset.tex",
        axis_height="\\winedatasetdiagramheight",
        axis_width="\\winedatasetdiagramwidth",
    )
else:
    plt.show()
# plt.show()
# print(ks_test)
# pickle.dump(
# favard_predictive_densities,
# open("favard_formula1_predictive_densities.p", "wb"),
# )
# pickle.dump(
# mercer_predictive_densities,
# open("mercer_formula1_predictive_densities.p", "wb"),
# )
