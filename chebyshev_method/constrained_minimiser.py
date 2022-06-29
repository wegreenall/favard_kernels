import torch
import scipy
from scipy import optimize
import torch.distributions as D
import numpy as np

catalan_numbers = np.array(
    [
        1,
        0,
        1,
        0,
        2,
        0,
        5,
        0,
        14,
        0,
        42,
        0,
        132,
        0,
        429,
        0,
        1430,
        0,
        4862,
        0,
        16796,
        0,
        58786,
        0,
        208012,
        0,
        742900,
        0,
        2674440,
        0,
        9694845,
        0,
        35357670,
        0,
        129644790,
        0,
        477638700,
        0,
        1767263190,
        0,
        6564120420,
        0,
    ]
)


order = 10
moments = catalan_numbers[1 : order + 1]
true_moment_matrix = torch.Tensor(catalan_numbers[1 : 2 * order + 1]).unfold(
    0, order, 1
)[1:, :]
ratios_matrix = np.einsum(
    "i, j -> ij",
    1 / np.linspace(2, order + 1, order),
    np.linspace(1, order, order),
)
# true_moment_matrix_inverse = torch.inverse(true_moment_matrix * ratios_matrix).numpy()
# breakpoint()
true_moment_matrix_inverse = torch.inverse(
    true_moment_matrix * ratios_matrix
).numpy()
scale_parameter = 1
le2 = np.zeros(order)
le2[1] = scale_parameter


def constraint_function(candidate_moments):
    first_moments = torch.Tensor(
        candidate_moments
    )  # tensor version of moments
    # print(first_moments)
    true_moments = catalan_numbers[order + 1 : 2 * order + 1]  # true moments
    full_moments = torch.cat((first_moments, torch.Tensor(true_moments)))
    moment_matrix = full_moments.unfold(0, order, 1)[1:, :].numpy()
    # breakpoint()
    moment_matrix_inverse = np.linalg.inv(
        moment_matrix * ratios_matrix
    )  # the variable inverse_matrix
    return (
        moment_matrix_inverse @ candidate_moments
        - true_moment_matrix_inverse @ moments
        + le2
    )
    # return np.zeros(order)


def objective(candidate_moments, *args):
    moments = args[0]
    return scipy.linalg.norm(candidate_moments - moments, 2)


constraint = optimize.NonlinearConstraint(
    constraint_function, np.zeros(order), np.zeros(order)
)
result = optimize.minimize(
    objective,
    moments,
    (moments,),
    constraints=(constraint),
    method="trust-constr",
)

print("calculated_moments", result["x"])
print("True moments:", moments)
print(np.linalg.norm(result["x"] - moments))
