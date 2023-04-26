import torch
import torch.distributions as D
import matplotlib.pyplot as plt


def test_function(x: torch.Tensor) -> torch.Tensor:
    """
    Test function used in an iteration of Daskalakis, Dellaportas and Panos.
    """
    return (1.5 * torch.sin(x) + 0.5 * torch.cos(4 * x) + x / 8).squeeze()


def weight_function(input: torch.Tensor):
    return torch.exp(-(input ** 2) / 4)


def get_training_inputs(sample_shape):
    """
    This function generates a sample from a Gaussian distribution
    and a non-Gaussian distribution. The corresponding samples
    are used as the inputs for Gaussian process models.
    Ostensibly using the Gaussian inputs with the standard Gaussian Mercer
    GP should work better than the same Mercer GP with non-Gaussian inputs.
    """
    # input distributions
    # Gaussian
    gaussian_input_distribution = D.Normal(
        torch.Tensor([0.0]), torch.Tensor([1.0])
    )

    # non-Gaussian
    mixing_distribution = D.Categorical(torch.Tensor([0.2, 0.8]))
    component_distribution = D.Normal(
        torch.Tensor([-2.7, 2.7]), torch.Tensor([0.6, 0.6])
    )

    mixture_input_distribution = D.MixtureSameFamily(
        mixing_distribution, component_distribution
    )

    # get samples
    gaussian_input_sample = gaussian_input_distribution.sample(sample_shape)
    non_gaussian_input_sample = mixture_input_distribution.sample(sample_shape)
    return gaussian_input_sample, non_gaussian_input_sample
