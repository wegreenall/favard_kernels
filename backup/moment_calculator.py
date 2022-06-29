import torch

import torch.distributions as D
from termcolor import colored

torch.set_printoptions(linewidth=200, precision=20)
"""
This program will be used to test and initially implement the mechanism by which
we choose the moments for the orthonormal basis.

We want to build an orthonormal basis φ_i so that 
        \int φ_i φ_j dΨ  = δ_ij

where Ψ represents the input measure.

The basis function will be written:
     φ_i = c_i P_i √w̅(x)

Which means that the integral above is written:
        \int c_i c_j P_i P_j w(x) dΨ  = δ_ij

For this to hold true, the sequence of orthogonal polynomials
                        {P_i}
must be orthogonal w.r.t the product weight function 
                h(x) === w(x)ψ(x) 

where we write ψ(x) as the density associated wtih the measure Ψ(x);
i.e. we assume Ψ to be absolutely continuous.

However, we only have the moments from the input measure Ψ as calculated from the sample;
we do not have the moments from the combined density h(x).

To get these, we need to select a set of moments "close" to the moments
from the input measure.

We can calculate the moments from the input measure and assume ψ(x) to be the
maximal entropy density under the moment constraints matching the input:

This has the form of an exponentiated polynomial, with coefficients the solution
to the system:
                    μ = Μδ
These δ are the Lagrange multipliers for the maximal entropy density problem
with moment constraints.

We choose the weight function w(x) to be e^{-lx^2} - this gives "nice" behaviour
to the basis functions. This can also be done with e^{-l|x|} if so desired.

The resulting product weight is therefore the maximal entropy density for a
sequence of Lagrange multipliers δ + le₂. There will be a "ring" around the
solution δ of feasible moments that could produce these Lagrange multiplers.

Assuming the input measure to have the maximal entropy density described above,
we can construct a sequence of moments in a principled fashion. To see this,
take:

            argmin (μ̂) ||μ̂ - μ|| s.t. M̂⁻¹μ̂ = Μ⁻¹μ + le₂

where l is the weight parameter on the weight function. This essentially says
that we use as our moments that vector of moments that is closest to the true
ones, given that the Lagrange multipliers from the corresponding optimisation
problem are correct.

Notice that inside M̂⁻¹ we have 2m moments; the latter m moments are considered
to be the same as the true moments. Hence, we only have to choose the vector 
μ̂. The problem above does not have a linear constraint; the optimisation
problem above however only needs to be solved once (and should be relatively 
quick given the convexity of the objective). As a result we will solve this
problem once for the input data to construct the "close" moments for the
orthogonal polynomial sequence, and then use the resulting orthogonal
polynomial basis to construct the Favard kernel.

"""


def moment_matrix(input_moments: torch.Tensor, order) -> torch.Tensor:
    """
    Returns the Hankel matrix for a given sequence of moments
    in param input_moments. It is necessary that this sequence of moments
    be 2 * m so that the size of the Hankel matrix is correct.
    """
    assert len(input_moments) == 2 * order
    moment_matrix = input_moments.unfold(0, order, 1)[1:, :]  # moment matrix
    return moment_matrix


def get_deltas(input_moments: torch.Tensor, order: int) -> torch.Tensor:
    """
    Returns the solution to the system
                m = M δ

    for the choice of "close" moments.
    """
    # ratios_matrix R_{ij} = j/(i+1)
    ratios_matrix = torch.einsum(
        "i, j -> ij",
        1 / torch.linspace(2, order + 1, order),
        torch.linspace(1, order, order),
    )
    true_moment_matrix = moment_matrix(input_moments, order)
    true_moment_vector = input_moments[:order]
    system_matrix = ratios_matrix * true_moment_matrix
    solution = torch.linalg.solve(system_matrix, true_moment_vector)
    return solution


def lagrangian(
    mus: torch.Tensor,
    input_moments: torch.Tensor,
    constraint_variable: torch.Tensor,
    lagmul: torch.Tensor,
    order: int,
    iteration: int,
):
    """
    Returns the Lagrangian, evaluated at the current iteration (passed in) of
    mus:
            L(μ̂) = ||μ̂ - μ|| + λ(M̂⁻¹μ̂ - (Μ⁻¹μ + le₂))

    In the initial phase, I will write this to just solve to minimise the
    distance between μ and μ̂.

    param mus: What is being trained
    param input_moments: the full 2m true moments
    """
    obj_term = torch.norm(mus - input_moments[:order])

    # now build the constraint term
    mus_soln = get_deltas(torch.cat((mus, input_moments[order:])), order)
    constraint_term = lagmul @ (constraint_variable - mus_soln)
    with torch.no_grad():
        if iteration % 1000 == 0:
            print("######\n")
            # print(colored("M̂⁻¹μ̂ : ", "magenta"), colored(mus_soln, "red"))
            # print(
            # colored("constraint_variable: ", "yellow"),
            # colored(constraint_variable, "green"),
            # )
            print(colored("CONSTRAINT TERM:", "yellow"), constraint_term)
            print("#######\n")
    return obj_term + constraint_term


def get_close_moments(
    input_moments: torch.Tensor, length_scale: torch.Tensor, order
) -> torch.Tensor:
    """
    Given the input moments, constructs the "close" moments such that
    the Lagrange multipliers are shifted. The result is that the OPS
    we construct will be orthogonal w.r.t the "close moments" - and the weight
    function will be nearly correct.

    (I have a feeling that they would get the same answer...)
    """
    close_moments = torch.Tensor([order])
    assert (
        len(input_moments) == 2 * order
    ), "Please make sure there are 2 * order moments"

    # set up parameters for the optimisation
    convergence_eps = 0.001

    # optimisables:
    mus = 1.1 * input_moments[:order].clone()
    # initially, let's set the mus to 0 so we can see how it works.
    # mus = torch.zeros([order]).unsqueeze(0)

    lagmul = 0.0003 * torch.ones(order)
    mus.requires_grad = True
    lagmul.requires_grad = True

    # constraint
    le2 = torch.zeros(order)
    # le2[1] = length_scale
    deltas = get_deltas(input_moments, order)
    constraint_variable = deltas + le2

    # optimisation setup
    lagmul_optimiser = torch.optim.SGD([lagmul], lr=0.5)

    # begin the optimisation
    convergence_criterion = False
    mus_converged = False
    lagmul_converged = False

    iterations = 0
    while not lagmul_converged:
        mus_optimiser = torch.optim.SGD([mus], lr=0.5)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            mus_optimiser, gamma=0.5
        )
        mus_iterations = 0
        while not mus_converged and (mus_iterations < 200000):
            mus_iterations += 1
            mus_optimiser.zero_grad()
            lagmul_optimiser.zero_grad()
            output = lagrangian(
                mus,
                input_moments,
                constraint_variable,
                lagmul,
                order,
                mus_iterations,
            )
            output.backward()
            mus_optimiser.step()
            mus_converged = (torch.abs(mus.grad) < convergence_eps).all()
            if mus_iterations % 1000 == 0:
                print("#######\n")
                print("Current ", colored("mus:", "red"), mus)
                print("#######\n")
                # print("Current mus grad:", colored(mus.grad, "blue"))
                print("Current ", colored("mus grad:", "blue"), mus.grad)
                print("#######\n")
                print(
                    colored("Current lagmul:", "green"),
                    colored(lagmul, "cyan"),
                )
                print(colored("Current lagmul grad:", "green"), lagmul.grad)
                scheduler.step()
        iterations += mus_iterations
        if mus_converged:
            print("Mus converged!")
        else:
            print("Mus NOT converged!")

        print("Scheduler_stepped")
        print(mus_optimiser)
        print("Current lagmul:", lagmul)
        print(colored("Current lagmul grad:", "green"), lagmul.grad)
        # scheduler.step()
        lagmul_optimiser.zero_grad()
        lagmul_output = lagrangian(
            mus, input_moments, constraint_variable, lagmul, order, iterations
        )
        lagmul_output.backward(
            torch.tensor(
                -1,
            )
        )
        # lagmul_output.backward()
        lagmul_optimiser.step()

        # lagmul_converged = (torch.abs(lagmul.grad) < convergence_eps).all()
        lagmul_converged = True
        print("FINAL ITERATIONS:", iterations)
    return mus


if __name__ == "__main__":
    print("Running!")
    length_scale = torch.tensor(
        5,
    )
    order = 8
    catalans = torch.Tensor(
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
    input_moments = catalans[: 2 * order]
    final_moments = get_close_moments(input_moments, length_scale, order)
    print("Final solved moments:", final_moments)
