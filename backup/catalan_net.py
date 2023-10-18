import torch

from torch.nn import Module, Linear
from torch import nn

"""
Due to Aigner (200?) there is a view of the orthogonal polynomial recursion
used in Favard's theorem that corresponds to a graph; the graph's weights
are built from the betas and gammas in the OPS recursion.

The weight of a node in the graph is calculated as the sum of the weights
of the paths leading into the node; the weight of a path is the product of the 
weights of the edges along the path.
The moments of the corresponding linear functional are the values of the nodes
in the first column  in the graph, constructed by adding all weights of the
paths leading from node (0,0) into the given node.
"""


def jordan_matrix(s, t):
    order = len(s)
    ones = torch.ones(order)
    matrix = torch.zeros((order + 1, order))
    block = torch.vstack((torch.ones(order), s, t)).t()
    for i in range(order + 1):
        # breakpoint()
        if i < order - 2:
            matrix[i, i : i + 3] = torch.Tensor([ones[i], s[i], t[i]])
        elif i < order - 1:
            matrix[i, i : i + 2] = torch.Tensor([ones[i], s[i]])
        elif i < order:
            matrix[i, i] = torch.Tensor([ones[i]])
    # for i
    # print(matrix)
    # breakpoint()
    return matrix[:, :].t()


class CatNet(nn.Module):
    def __init__(self, order):
        super().__init__()
        s = 0 * torch.ones(2 * order)
        t = 1 * torch.ones(2 * order)
        jordan = jordan_matrix(s, t)  # .t()
        self.order = order
        breakpoint()
        self.layers = []
        for i in range(1, order + 1):
            layer = nn.Linear(
                i, i + 1, bias=False
            )  #  for i in range(1, order + 1)

            # breakpoint()
            with torch.no_grad():
                if i < order - 1:
                    try:
                        layer.weight[:, :] = jordan[:i, 1 : i + 2].t()
                    except RuntimeError:
                        print("Run time error!")
                        breakpoint()
                elif i < order:
                    # print(i)
                    # breakpoint()
                    layer.weight[:, :] = jordan[:i, :].t()
                else:
                    layer.weight[:, :] = jordan[: order + 1, : order + 1]

            self.layers.append(layer)
        self.cat = nn.Sequential(*self.layers)
        pass

    def forward(self, x):
        return_val = torch.zeros(self.order)
        for i, layer in enumerate(self.layers[:order]):
            x = layer(x)
            return_val[i] = x[0]
        # breakpoint()
        return return_val


class CatNet2(nn.Module):
    def __init__(self, order):
        super().__init__()
        self.order = order
        s = torch.linspace(1, order, order)
        # s[0] = 2
        t = torch.linspace(0.5, order - 0.5, order)
        print("s = ", s, "\nt = ", t)
        s = 0 * torch.ones(1 * order)
        t = torch.ones(1 * order)
        jordan = torch.nn.Parameter(jordan_matrix(s, t))  # .t()
        print(jordan)
        self.layers = []
        self.shared_weights = []
        for i in range(1, order + 1):
            self.shared_weights.append(torch.nn.Parameter(jordan[: i + 1, :i]))
            layer = nn.Linear(i, i + 1, bias=False)
            self.layers.append(layer)
        return

    def forward(self, x):
        """
        As is, with s = 0 and t = 1, the resulting values should be the Catalan
        numbers. The parameters need to be shared between all the layers...

        This needs to be done by sourcing the parameters from the Jordan matrix
        but i'm not sure that this is what is happening here.
        """
        catalans = torch.zeros(self.order)
        for i, layer in enumerate(self.layers):
            layer.weight = self.shared_weights[
                i
            ]  # is this a reference or value?
            x = layer(x)
            catalans[i] = x[
                -1
            ]  # the last in the layer will be the corresponding catalan no.
        return catalans


if __name__ == "__main__":
    order = 8
    my_net = CatNet2(order)
    my_net(torch.Tensor([1.0]))
    optimiser = torch.optim.SGD(my_net.shared_weights, 0.001)
    catalans = my_net(torch.Tensor([1.0]))
    orig_catalans = catalans.clone()
    # s = 2 * torch.ones(order)
    # t = 3 * torch.ones(order)
    # jordan = jordan_matrix(s, t)
    print(my_net.shared_weights)
    # optimisation stuff
    catalans.backward(torch.ones(catalans.shape))
    optimiser.step()
    catalans2 = my_net(torch.Tensor([1.0]))
    print("Original catalans:", orig_catalans)
    print("Catalans2:", catalans2)
    print(my_net.shared_weights)
