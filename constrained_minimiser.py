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
moments = catalan_numbers[:order]
true_moment_matrix = torch.unfold(0, catalan_numbers[: 2 * order], 1)
true_moment_matrix_inverse = torch.inverse(true_moment_matrix)

# def get_moments

def constraint_function(candidate_moments):
    first_moments = torch.Tensor(candidate_moments)  # tensor version of moments
    true_moments = catalan_numbers[order : 2 * order]  # true moments
    moment_matrix = torch.unfold(0, torch.cat((first_moments, true_moments)), 1)
    moment_matrix_inverse = 
    return


def objective(candidate_moments, *args):
    moments = args[0]
    return scipy.linalg.norm(candidate_moments - moments, 2)


result = optimize.minimize(objective, moments + np.ones(order), (moments,))
constraint = scipy.optimiser.NonLinearConstraint(constraint_function, 0, 0)

print(result)
