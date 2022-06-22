import torch
from ortho.orthopoly import OrthonormalPolynomial
from ortho.builders import get_orthonormal_basis_from_sample
from ortho.measure import MaximalEntropyDensity
from ortho.basis_functions import OrthonormalBasis
from mercergp.MGP import MercerKernel, MercerGP
from mercergp.eigenvalue_gen import EigenvalueGenerator
from mercergp.likelihood import MercerLikelihood
import matplotlib.pyplot as plt
from typing import Callable


def train_favard_params(
    parameters: dict,
    eigenvalue_generator: EigenvalueGenerator,
    order: int,
    input_sample: torch.Tensor,
    output_sample: torch.Tensor,
    weight_function: Callable,
    optimiser: torch.optim.Optimizer,
    dim=1,
) -> dict:

    basis = get_orthonormal_basis_from_sample(
        input_sample, weight_function, order
    )
    mgp_likelihood = MercerLikelihood(
        order,
        optimiser,
        basis,
        input_sample,
        output_sample,
        eigenvalue_generator,
    )
    new_parameters = parameters.copy()
    mgp_likelihood.fit(new_parameters)
    return new_parameters


def build_favard_gp(
    parameters: dict,
    eigenvalue_generator: EigenvalueGenerator,
    order: int,
    input_sample: torch.Tensor,
    output_sample: torch.Tensor,
    weight_function: Callable,
    dim=1,
) -> MercerGP:
    """
    Returns a Mercer Gaussian process with a basis constructed from the input
    sample measure.

    param eigenvalue_generator: a callable that returns, a tensor of "order"
            eigenvalues, with input being the parameter dictionary 'parameters'
    """
    # get the corresponding orthonormal basis.
    # weight_function
    basis = get_orthonormal_basis_from_sample(
        input_sample, weight_function, order
    )
    eigenvalues = eigenvalue_generator(parameters)

    # build the kernel
    kernel = MercerKernel(order, basis, eigenvalues, parameters)

    # build the gp
    mgp = MercerGP(basis, order, dim, kernel)
    return mgp
