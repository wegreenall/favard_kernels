from ortho.builders import get_gammas_from_sample
import numpy as np
import torch
import torch.distributions as D
import tabulate
from scipy.stats import semicircular as wigner_distribution

"""
This script will calculate, N times, the orthogonal polynomial coefficients
for the Normal distribution. Then we will present the mean, and variance, of the 
corresponding recurrence coefficients.
"""


class scipywrapper(object):
    def __init__(self, dist):
        self.dist = dist

    def sample(self, size):
        # breakpoint()
        return torch.Tensor(self.dist.rvs(size=size))


def gammas_means_and_stds(distribution, sample_size, N, order, truth):
    gammas = torch.Tensor(N, order)

    for i in range(N):
        sample = distribution.sample((sample_size,))
        next_gammas = get_gammas_from_sample(sample, order)
        gammas[i, :] = next_gammas
    means = torch.mean(gammas, dim=0)
    variances = torch.std(gammas, dim=0)
    # print(means)
    # print(variances)
    data = torch.vstack(
        (torch.Tensor(list(range(order))), means, variances, truth)
    )

    return data


normal_distribution = D.Normal(0, 1)
dry_run = False
if dry_run:
    sample_size = 1000
    N = 50
else:
    sample_size = 100000
    N = 5000
order = 6
normal_truth = torch.Tensor([1, 1, 2, 3, 4, 5])
wigner_truth = torch.Tensor([1, 0.25, 0.25, 0.25, 0.25, 0.25])

normal_data = gammas_means_and_stds(
    normal_distribution, sample_size, N, order, normal_truth
)
wigner_distribution = scipywrapper(wigner_distribution)
wigner_data = gammas_means_and_stds(
    wigner_distribution, sample_size, N, order, wigner_truth
)
# breakpoint()
normal_table = tabulate.tabulate(
    normal_data.t(),
    headers=("Order", "Means", "Standard Deviation", "Ground Truth"),
    tablefmt="simple",
)
wigner_table = tabulate.tabulate(
    wigner_data.t(),
    headers=("Order", "Means", "Standard Deviation", "Ground Truth"),
    tablefmt="simple",
)

print(normal_table)
print(wigner_table)
file = open("normal_stability.mk2", "w")
file.write(normal_table)
file.close()

file = open("wigner_stability.mk2", "w")
file.write(wigner_table)
file.close()
# Now calculate the same thing for the Chebyshev polynomials.
