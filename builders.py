import torch
from ortho.orthopoly import OrthonormalPolynomial
from ortho.measure import MaximalEntropyDensity
from ortho.basis_functions import OrthonormalBasis
from mercergp.MGP import MercerKernel, MercerGP


def build_favard_gp(parameters: dict, order: int):
    """
    Returns a Mercer GP, given the training of a likelihood.

    param parameters: a dictionary containing the parameters for the
                      Mercer Likelihood
    """
    gammas = parameters["gammas"].detach().clone()
    noise_parameters = parameters["noise_parameter"].detach().clone()
    eigenvalue_smoothness_parameter = (
        parameters["eigenvalue_smoothness_parameter"].detach().clone()
    )
    eigenvalue_scale_parameter = (
        parameters["eigenvalue_scale_parameter"].detach().clone()
    )
    shape_parameter = parameters["shape_parameter"].detach().clone()

    betas = torch.zeros(2 * order)
    final_orthopoly = OrthonormalPolynomial(
        order, betas[:order], gammas[:order]
    )
    final_weight_function = MaximalEntropyDensity(order, betas, gammas)
    final_basis = OrthonormalBasis(
        final_orthopoly, final_weight_function, 1, order
    )

    eigenvalues = (
        eigenvalue_scale_parameter
        / (torch.linspace(1, order, order) + shape_parameter)
        ** eigenvalue_smoothness_parameter
    )
    kernel_params = {"noise_parameter": parameters["noise_parameter"]}
    kernel = MercerKernel(order, final_basis, eigenvalues, kernel_params)
    return MercerGP(final_basis, order, 1, kernel)
