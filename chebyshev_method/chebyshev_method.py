import torch
import torch.distributions as D
from torch import nn
from termcolor import colored
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=10, linewidth=200)

"""
The Gautschi technique via the Chebyshev method appears to have a problem of 
poor conditioning. The construction of the map

        m -> ρ
where ρ are the recurrence coefficients in the Favard recursion is apparently 
poorly conditioned. This is decomposed by Gautschi (1982) into the
composition of two maps
        H ∘ G
    H : m -> γ
    G : γ -> ρ
Gautschi explains that H is well conditioned (see Gautschi, 1986) but that 
G is not. We aim to test whether it is feasible to use optimisation techniques
that would not have been easy to implement at that time to improve on errors.
"""


def weight_mask(order) -> (torch.Tensor, torch.Tensor):
    # order = len(s)
    ones = torch.ones(order)
    zeros = torch.zeros(order)
    matrix = torch.zeros((order + 1, order))
    matrix_2 = torch.zeros((order + 1, order))

    # block = torch.vstack((torch.ones(order), s, t)).t()
    for i in range(order + 1):
        # breakpoint()
        if i < order - 2:
            matrix[i, i : i + 3] = torch.tensor([zeros[i], ones[i], ones[i]])
            matrix_2[i, i : i + 3] = torch.tensor(
                [ones[i], zeros[i], zeros[i]]
            )
        elif i < order - 1:
            matrix[i, i : i + 2] = torch.tensor([zeros[i], ones[i]])
            matrix_2[i, i : i + 2] = torch.tensor([ones[i], zeros[i]])
        elif i < order:
            matrix[i, i] = torch.tensor([zeros[i]])
    return matrix[:, :].t(), matrix_2[:, :].t()


def jordan_matrix(s, t):
    """
    Generates the Jordan matrix from which weight views will be constructed.
    In most of the examples, s is β, t is γ.
    """
    order = len(s)
    ones = torch.ones(order)
    zeros = torch.zeros(order)
    matrix = torch.zeros((order + 1, order))
    block = torch.vstack((torch.ones(order), s, t)).t()
    for i in range(order + 1):
        # breakpoint()
        if i < order - 2:
            matrix[i, i : i + 3] = torch.tensor([ones[i], s[i], t[i]])
        elif i < order - 1:
            matrix[i, i : i + 2] = torch.tensor([ones[i], s[i]])
        elif i < order:
            matrix[i, i] = torch.tensor([ones[i]])
    # for i
    # print(matrix)
    # breakpoint()
    return matrix[:, :].t()


class CatNet(nn.Module):
    """
    A Catalan matrix represented as a Neural net, to allow
    us to differentiate w.r.t. its parameters easily.
    """

    def __init__(self, order, betas, gammas):
        super().__init__()

        # Check the right size of the initial s
        if (betas.shape[0] != 2 * order) or (gammas.shape[0] != 2 * order):
            raise ValueError(
                r"Please provide at least 2 * order parameters for beta and gamma"
            )
        self.order = order
        # self.mask, self.ones_matrix = weight_mask(2 * order)
        self.jordan = jordan_matrix(betas, gammas)  # .t()
        self.mask_matrix = jordan_matrix(
            torch.ones(2 * order), torch.ones(2 * order)
        )
        breakpoint()
        self.layers = []

        for i in range(1, 2 * order + 1):
            layer = nn.Linear(i, i + 1, bias=False)
            layer.weight = torch.nn.Parameter(
                self.jordan[: i + 1, :i].view(-1, i)
            )
            self.layers.append(layer)
        return

    def mask_jordan(self):
        """
        Applies the mask so that no values are placed outside the 3-term
        recursion structure of the graph.
        """
        with torch.no_grad():
            self.jordan.fill_diagonal_(1.0)
            self.jordan *= self.mask_matrix

    def forward(self, x):
        """
        As is, with s = 0 and t = 1, the resulting values should be the Catalan
        numbers. The parameters need to be shared between all the layers...

        This needs to be done by sourcing the parameters from the Jordan matrix
        but i'm not sure that this is what is happening here.
        """
        # first, mask the weights that should be 0
        self.mask_jordan()

        # build the sequence of moments
        candidate_moments = torch.zeros(2 * self.order)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            candidate_moments[i] = x[-1]

        return torch.hstack(
            (
                torch.tensor(
                    1,
                ),
                candidate_moments[:-1],
            )
        )


if __name__ == "__main__":
    order = 6
    opt_norm = False
    # betas = torch.zeros(2 * order)
    betas = torch.ones(2 * order)
    gammas = torch.ones(2 * order)
    betas.requires_grad = True
    gammas.requires_grad = True
    cat_net = CatNet(order, betas, gammas)
    value = cat_net(torch.Tensor([1.0]))
    print(value)

    initial_jordan = jordan_matrix(betas, gammas)
    correct_jordan = jordan_matrix(
        torch.zeros(2 * order), torch.ones(2 * order)
    )
    for i in range(1, 2 * order + 1):
        weights = initial_jordan[: i + 1, :i].view(-1, i)
        # print(weights)
    true_moments = torch.Tensor(
        [1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 5.0, 0.0, 14.0, 0.0]
    )
    start_moments = torch.zeros(true_moments.shape)

    # set up optimisation
    breakpoint()

    params_list = []
    for layer in cat_net.layers:
        layer_params = [param for param in layer.parameters()]
        params_list.extend(layer_params)

    if opt_norm:
        # optimiser = torch.optim.Adam(params_list, lr=0.000005)
        optimiser = torch.optim.Adamax(params_list, lr=0.00005)

        # optimiser = torch.optim.SGD(params_list, lr=0.00005)
    else:
        optimiser = torch.optim.Adamax(params_list, lr=0.00005)
        # optimiser = torch.optim.Adam(params_list, lr=0.000005)

        # optimiser = torch.optim.SGD(params_list, lr=0.00005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.99)

    iterations = 0
    while not torch.allclose(start_moments, true_moments, atol=1e-13):
        optimiser.zero_grad()
        start_moments = cat_net(torch.Tensor([1.0]))
        if opt_norm:
            loss = torch.norm(true_moments - start_moments, 2)
            loss.backward()
        else:
            loss = torch.pow(true_moments - start_moments, 2)
            loss.backward(torch.ones(2 * order))
        # with torch.no_grad():
        # cat_net.
        optimiser.step()
        with torch.no_grad():
            cat_net.mask_jordan()
        if iterations % 500 == 0:
            print(true_moments - start_moments)
            betas_gammas_jordan = [
                param for param in cat_net.layers[-1].parameters()
            ][0]
            # breakpoint()
            print(betas_gammas_jordan - correct_jordan[:, :-1])
            # print(betas_gammas_jordan)
        iterations += 1
        if iterations % 50000 == 0:
            scheduler.step()
            print(colored("##################", "red"))
            print(colored("SCHEDULER STEPPED!", "red"))
            print(colored("##################", "red"))

    print("COMPLETED!")
