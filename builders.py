import torch
import torch.autograd as autograd
from ortho.orthopoly import OrthonormalPolynomial
from ortho.builders import OrthoBuilder
from ortho.measure import MaximalEntropyDensity
from ortho.basis_functions import OrthonormalBasis
from mercergp.MGP import MercerKernel, MercerGP
from mercergp.eigenvalue_gen import EigenvalueGenerator, FavardEigenvalues
from mercergp.likelihood import MercerLikelihood
import matplotlib.pyplot as plt
from typing import Callable


def train_favard_params(
    parameters: dict,
    # eigenvalue_generator: EigenvalueGenerator,
    order: int,
    input_sample: torch.Tensor,
    output_sample: torch.Tensor,
    weight_function: Callable,
    optimiser: torch.optim.Optimizer,
    dim=1,
) -> dict:
    """ """

    basis = (
        OrthoBuilder(order)
        .set_sample(input_sample)
        .set_weight_function(weight_function)
        .get_orthonormal_basis()
    )

    # x = torch.tensor(0.0)
    x = torch.Tensor([0.0])
    x.requires_grad = True

    # get the basis at zero and its derivatives
    f0 = basis(x)
    # breakpoint()
    df0 = autograd.functional.jacobian(basis, x).squeeze(2)
    d2f0 = f0.clone()  # autograd.grad(df0, x)[0]
    breakpoint()
    assert (
        f0.shape == df0.shape == d2f0.shape
    ), "derivative and second derivative are the wrong shape : ASSRT STATEMENT"

    # build the Favard eigenvalue generator
    eigenvalue_generator = FavardEigenvalues(order, f0, df0, d2f0)

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
    # for param in filter(
    # lambda param: (isinstance(param, torch.Tensor))
    # and (param.requires_grad),
    # new_parameters.values(),
    # ):
    # # new_parameters[param] = new_parameters[param].detach()
    # param = param.detach()
    for param in filter(
        lambda param: isinstance(new_parameters[param], torch.Tensor),
        new_parameters,
    ):
        print(new_parameters[param])
        new_parameters[param] = new_parameters[param].detach()

    return new_parameters


def build_favard_gp(
    parameters: dict,
    order: int,
    input_sample: torch.Tensor,
    weight_function: Callable,
    eigenvalue_generator: EigenvalueGenerator,
    dim=1,
) -> MercerGP:
    """
    Returns a Mercer Gaussian process with a basis constructed from the input
    sample measure.

    param eigenvalue_generator: a callable that returns, a tensor of "order"
            eigenvalues, with input being the parameter dictionary 'parameters'
    param weight_function: a function w(x). When constructing the basis,
                           will have the square root applied to get w^{1/2}
    """
    # get the corresponding orthonormal basis.
    # weight_function
    # basis = get_orthonormal_basis_from_sample(
    # input_sample, weight_function, order
    # )
    basis = (
        OrthoBuilder(order)
        .set_sample(input_sample)
        .set_weight_function(weight_function)
        .get_orthonormal_basis()
    )
    eigenvalues = eigenvalue_generator(parameters)

    # build the kernel
    kernel = MercerKernel(order, basis, eigenvalues, parameters)

    # build the gp
    mgp = MercerGP(basis, order, dim, kernel)
    # breakpoint()
    return mgp
