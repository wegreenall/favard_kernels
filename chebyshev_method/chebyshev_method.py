import torch
import torch.distributions as D
from torch import nn
from termcolor import colored
import matplotlib.pyplot as plt
import ortho.builders as builders

torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=5, linewidth=300)

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

torch.manual_seed(1)


def weight_mask(order) -> (torch.Tensor, torch.Tensor):
    # order = len(s)
    ones = torch.ones(order)
    zeros = torch.zeros(order)
    matrix = torch.zeros((order + 1, order))
    matrix_2 = torch.zeros((order + 1, order))

    # block = torch.vstack((torch.ones(order), s, t)).t()
    for i in range(order + 1):
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
        if i < order - 2:
            matrix[i, i : i + 3] = torch.tensor([ones[i], s[i], t[i]])
        elif i < order - 1:
            matrix[i, i : i + 2] = torch.tensor([ones[i], s[i]])
        elif i < order:
            matrix[i, i] = torch.tensor([ones[i]])
    # for i
    print("JORDAN MATRIX:")
    print(colored(matrix.t(), "red"))
    # breakpoint()
    return matrix[:, :].t()


def jordan_matrix_fixed(s, t):
    """
    Generates the Jordan matrix from which weight views will be constructed.
    In most of the examples, s is β, t is γ.
    """
    order = len(s)
    ones = torch.ones(order)
    zeros = torch.zeros(order)
    matrix = torch.zeros((order + 1, order + 1))
    block = torch.vstack((torch.ones(order), s, t)).t()
    matrix[-2, order] = 1.0
    matrix[-1, order] = s[0]
    for i in range(order - 2, -1, -1):
        print(matrix)
        if i == order - 2:
            matrix[i : i + 3, i + 1] = torch.tensor(
                [ones[order - i - 2], s[order - i - 1], t[order - i - 2]]
            )
        elif i < order - 2:
            # print("Aboutt obuild the penultimate column!")
            # breakpoint()
            matrix[i : i + 3, i + 1] = torch.tensor(
                [ones[order - i - 2], s[order - i - 1], t[order - i - 2]]
            )
            # print("matrix after edit", order - i),
            # print(matrix)

    # print("JORDAN MATRIX FIXED:")
    # print(colored(matrix[:, 1:], "red"))
    # breakpoint()
    return matrix[:, 1:]


class CatNet(nn.Module):
    """
    A Catalan matrix represented as a Neural net, to allow
    us to differentiate w.r.t. its parameters easily.
    """

    def __init__(self, order, betas, gammas):
        super().__init__()

        # Check the right size of the initial s
        if betas.shape[0] == order and gammas.shape[0] == order:
            betas = torch.cat((betas, torch.ones(order)))
            gammas = torch.cat((gammas, torch.ones(order)))

        elif (betas.shape[0] != 2 * order) or (gammas.shape[0] != 2 * order):
            raise ValueError(
                r"Please provide at least 2 * order parameters for beta and gamma"
            )
        self.order = order
        # self.mask, self.ones_matrix = weight_mask(2 * order)
        self.jordan = jordan_matrix_fixed(betas, gammas)  # .t()
        self.mask_matrix = jordan_matrix_fixed(
            torch.ones(2 * order), torch.ones(2 * order)
        )
        # self.mask_matrix[:, 0] = torch.zeros(2 * order)
        # self.mask_matrix[0, 0] = 1.0
        # self.mask_matrix[1, 0] = 0.0
        print(colored(self.mask_matrix, "yellow"))
        print("\n")
        self.layers = []

        # for i in range(1, 2 * order + 1):
        for i in range(2 * order, 0, -1):
            layer = nn.Linear(2 * order - i + 1, 2 * order - i + 2, bias=False)
            this_jordan = self.jordan[i - 1 :, i - 1 :]
            these_weights = torch.nn.Parameter(
                this_jordan.view(-1, 2 * order - i + 1)
            )
            layer.weight = these_weights
            print("This layer", layer)
            print("This layer weight:", layer.weight)
            # breakpoint()
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
        # breakpoint()
        return torch.hstack(
            (
                torch.tensor(
                    1,
                ),
                candidate_moments[:-1],
            )
        )


if __name__ == "__main__":
    order = 14
    opt_norm = False
    opt_symmetric = True
    optimise_all = True
    # betas = torch.zeros(2 * order)
    betas = torch.ones(2 * order)
    gammas = torch.ones(2 * order)
    sampling_dist = D.Normal(1.0, 0.5)
    # betas = sampling_dist.sample((2 * order))
    # gammas = sampling_dist.sample((2 * order))
    true_moments = torch.Tensor(
        [
            1.0,
            0.0,
            1.0,
            0.0,
            2.0,
            0.0,
            5.0,
            0.0,
            14.0,
            0.0,
            42.0,
            0.0,
            132.0,
            0.0,
            429.0,
            0.0,
            1430.0,
            0.0,
            4862.0,
            0.0,
            16796.0,
            0.0,
            58786.0,
            0.0,
            208012.0,
            0.0,
            742900.0,
            0.0,
        ]
    )
    betas, gammas = builders.get_gammas_betas_from_moments(true_moments, order)
    betas.requires_grad = True
    gammas.requires_grad = True
    cat_net = CatNet(order, betas, gammas)
    random_input = sampling_dist.sample().unsqueeze(0)
    value = cat_net(torch.Tensor([1.0]))
    # value = cat_net(torch.Tensor(random_input))
    print(colored("Given initial betas", "blue"), colored(betas, "yellow"))
    print(colored("Given initial gammas", "blue"), colored(gammas, "yellow"))
    print("Calculated moments for given initial betas:", value)
    print(colored(torch.abs(true_moments - value), "green"))
    print("IDEA: use as initial betas the ones from the hankels")
    breakpoint()

    # initial_jordan = jordan_matrix(betas, gammas)
    # correct_jordan = jordan_matrix(
    # torch.zeros(2 * order), torch.ones(2 * order)
    # )
    # for i in range(1, 2 * order + 1):
    # weights = initial_jordan[: i + 1, :i].view(-1, i)
    # print(weights)
    if opt_symmetric:
        true_betas = torch.zeros(order)
        true_gammas = torch.ones(order)
        # true_moments = torch.Tensor(
        # [1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 5.0, 0.0, 14.0, 0.0, 42.0, 0.0]
        # )
    else:
        true_betas = torch.ones(order)
        true_gammas = torch.ones(order)
        true_moments = torch.Tensor(
            [
                1.0,
                1.0,
                2.0,
                4.0,
                9.0,
                21.0,
                51.0,
                127.0,
                323.0,
                835.0,
                2188.0,
                5798.0,
            ]
        )
    start_moments = torch.zeros(true_moments.shape)

    # set up optimisation

    if optimise_all:
        params_list = []
        for layer in cat_net.layers:
            layer_params = [param for param in layer.parameters()]
            params_list.extend(layer_params)
    else:  # i.e. try optimising with only the last layer's worth of parameters
        params_list = [param for param in cat_net.layers[order].parameters()]

    if opt_norm:
        # optimiser = torch.optim.Adam(params_list, lr=0.000005)
        # optimiser = torch.optim.Adamax(params_list, lr=0.00005)

        optimiser = torch.optim.SGD(params_list, lr=0.00005)
    else:
        # optimiser = torch.optim.Adam(params_list, lr=0.00000005)
        # optimiser = torch.optim.RMSprop(params_list, lr=0.00005)
        # optimiser = torch.optim.Adamax(params_list, lr=0.000005)

        optimiser = torch.optim.SGD(params_list, lr=0.000001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.999)

    iterations = 0
    while not torch.allclose(start_moments, true_moments):
        optimiser.zero_grad()
        # random_input = sampling_dist.sample().unsqueeze(0)
        start_moments = cat_net(torch.Tensor([1.0]))
        # start_moments = cat_net(random_input)
        if opt_norm:
            # breakpoint()
            loss = torch.norm(true_moments - start_moments, 4)
            loss.backward()
        else:
            loss = torch.pow(
                torch.abs(true_moments - start_moments), 4
            )  # + torch.norm(true_moments - start_moments, 2)
            loss.backward(torch.ones(2 * order))

        optimiser.step()
        with torch.no_grad():
            cat_net.mask_jordan()
        if iterations % 100 == 0:
            print(colored(true_moments - start_moments, "green"))
            betas_gammas_jordan = [
                param for param in cat_net.layers[-1].parameters()
            ][0]
            print(
                colored("true betas:", "yellow"), colored(true_betas, "yellow")
            )
            print(
                colored("truue gammas:", "blue"), colored(true_gammas, "blue")
            )
            # print(betas_gammas_jordan - correct_jordan[:, :-1])
            print(betas_gammas_jordan)
            # print(betas_gammas_jordan)
        iterations += 1
        if iterations % 50000 == 0:
            # scheduler.step()
            print(colored("##################", "red"))
            print(colored("SCHEDULER STEPPED!", "red"))
            print(colored("##################", "red"))

    print("COMPLETED!")
